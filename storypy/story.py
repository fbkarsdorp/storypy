import codecs
import operator
import os
import os.path as path
import sys

from collections import defaultdict, Counter
from itertools import combinations, groupby
from json import dumps

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import distance

from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from pattern.nl import sentiment as extract_sentiment, parse, Sentence
from dtw import dtw_distance
from HACluster import VNClusterer

sys.path.append(os.path.expanduser("~") + "/local/brat/server/src")
from annotation import Annotations
from ssplit import regex_sentence_boundary_gen

from storytiling import *
from storyplot import grid_plot
from utils import mean

# ----------------------------------------------------------------------------
# Functions to read the annotation files of Brat


def graphize(relations, entities):
    mentions = nx.DiGraph()
    for entity in sorted(entities, key=lambda e: entities[e].start):
        if entity in relations:
            for target in relations[entity]:
                mentions.add_edge(entity, target)
        else:
            mentions.add_edge(entity, entity)
    mentions = mentions.reverse()
    return mentions


def resolve(relations, entities):
    """Resolve the coreference chains."""
    mentions = graphize(relations, entities)
    for start_entity, _ in mentions.selfloop_edges():  # moet start er nog bij?
        chain = set(entity for edge in nx.bfs_edges(mentions, start_entity)
                    for entity in edge)
        yield chain if chain else set([start_entity])


def read_annotation_file(filename):
    """Loads a file annotated for co-references with brat."""
    annotations = Annotations(filename, read_only=True)
    entities = {entity.id: entity for entity in annotations.get_entities()
                if entity.type in ('Actor', 'Location')}
    relations = defaultdict(list)
    for relation in annotations.get_relations():
        relations[relation.arg1].append(relation.arg2)
    chains = resolve(relations, entities)
    actors, locations = [], []
    for chain in chains:
        chain = [(entity, entities[entity].tail.strip(),
                          entities[entity].get_start(),
                          entities[entity].get_end()) for entity in chain]
        entity_type = entities[chain[0][0]].type
        if entity_type == 'Actor':
            actors.append(chain)
        elif entity_type == 'Location':
            locations.append(chain)
    return actors, locations, entities

# -----------------------------------------------------------------------------
# Classes to create Story objects


class Entity(object):

    """An Entity represents either a character in a story or a location.
    Each entity consist of a list of all occurences in a story and a standardized
    name."""

    def __init__(self, i, chain):
        self.id = i
        self.chain = chain
        self.name = None
        self.cluster = None
        self.standardize()

    def standardize(self):
        """Standardize the chain mention of this Entity."""
        module_path = path.dirname(__file__)
        stopwords = set(w.strip()
                        for w in open(path.join(module_path, 'data', 'pronouns.txt')))
        try:
            longest_token, _ = Counter([token.lower() for _, token, _, _ in self.chain
                                        if token.lower() not in stopwords]).most_common()[0]
        except IndexError:
            _, longest_token = max(
                self.chain, key=lambda entity: len(entity[1]))
        longest_token += '-' + str(self.id)
        self.chain = [(id, longest_token, start, end) for id, _, start, end in self.chain]
        self.name = self.chain[0][1]

    def __eq__(self, other):
        if not isinstance(other, Entity):
            raise ValueError(
                "Can't compare to object of type: %s" % type(other))
        return self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return '<Entity(%s)>' % self.name

    def __str__(self):
        return self.name


class Scene(object):

    """A Scene represents is a part of a Story represented by a set
    of characters and a set of locations."""

    def __init__(self, start, end, characters=None, locations=None):
        if characters is not None:
            assert all(isinstance(character, Entity)
                       for character in characters)
            self.characters = characters
        else:
            self.characters = set()
        if locations is not None:
            assert all(isinstance(location, Entity) for location in locations)
            self.locations = locations
        else:
            self.locations = set()
        self.start, self.end = start, end
        self.sentiment = 0

    def distance(self, other, dist_between='characters'):
        """Compute the distance between this Scene and some other Scene
        on the basis of the Jaccard distance between their character or location sets
        or the union of their locations and characters."""
        if not isinstance(other, Scene):
            raise ValueError("Can't compare to %s" % type(other))
        if dist_between == 'characters':
            source, target = self.characters, other.characters
        elif dist_between == 'locations':
            source, target = self.locations, other.locations
        elif dist_between == 'both':
            source = self.characters.union(self.locations)
            target = other.characters.union(other.locations)
        else:
            raise ValueError(
                "The distance between %s cannot be computed." % dist_between)
        if not source or not other:
            return 1.0
        return distance.jaccard(source, target)

    def __eq__(self, other):
        if not isinstance(self, other):
            raise ValueError("Can't compare to %s" % type(other))
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return '<Scene(start=%d, end=%d)>' % (self.start, self.end)

    def merge(self, other):
        if not isinstance(other, Scene):
            raise ValueError("Can't merge with %s" % type(other))
        return Scene(min(self.start, other.start), max(self.end, other.end),
                     self.characters.union(other.characters),
                     self.locations.union(other.locations))

    def __str__(self):
        s = 'Scene (%s-%s; sentiment: %.3f):\n\n' % (
            self.start, self.end, self.sentiment)
        if self.characters:
            s += '   Characters:\n   -----------\n      %s\n\n' % ', '.join(
                map(str, self.characters))
        if self.locations:
            s += '   Locations:\n   ----------\n      %s\n\n' % ', '.join(
                map(str, self.locations))
        return s

    @staticmethod
    def concat(scenes):
        return reduce(operator.add, scenes)

    def __add__(self, other):
        return self.merge(other)


class Story(list):

    """A Story represents a sequence of Scene objects, where each scene is
    represented by a set of characters and a set of locations."""

    def __init__(self, id, filepath, scenes=[], characters=set(), locations=set()):
        assert all(isinstance(scene, Scene) for scene in scenes)
        list.__init__(self, scenes)
        self.id = id
        self.filepath = filepath
        self.characters = characters
        self.locations = locations

    def distance(self, other, entities='characters', constraint='none',
                 window=0, normalized=True, summed=False, sentiments=False):
        if not isinstance(other, Story):
            raise ValueError("Can't compare to %s" % type(other))
        source = self.to_dataframe(sentiments=sentiments, summed=summed).values
        target = other.to_dataframe(
            sentiments=sentiments, summed=summed).values
        return dtw_distance(source, target, constraint=constraint, window=window, step_pattern=2, normalized=normalized)

    def map(self, other, entities='characters', constraint='none', window=0, threshold=0.6):
        source, target = (self, other) if len(self.characters) > other.characters else (other, self)
        source_df = source.to_dataframe(entities=entities)
        target_df = target.to_dataframe(entities=entities)

        dm = np.zeros((len(source.characters), len(target.characters)))
        for i, character_i in enumerate(source_df.index):
            for j, character_j in enumerate(target_df.index):
                chars_s, chars_t = source_df.ix[i].values, target_df.ix[j].values
                if chars_s.shape[0] < chars_t.shape[0]:
                    chars_s, chars_t = chars_t, chars_s
                d = dtw_distance(chars_s, chars_t, constraint=constraint, window=window, normalized=True)
                dm[i, j] = d
        return pd.DataFrame(dm, index=source_df.index, columns=target_df.index)

    @staticmethod
    def load(filename, annotation_dir=None, text_dir=None):
        """Construct a new Story on the basis of some input file."""
        filepath, _ = os.path.splitext(filename)
        if annotation_dir != text_dir:
            annotation_file = os.path.join(annotation_dir, filepath)
            text_file = os.path.join(text_dirm, filepath)
        id = os.path.basename(filepath)
        characters, locations, entities = read_annotation_file(
            annotation_file + '.ann')
        characters = [Entity(i, character)
                      for i, character in enumerate(characters)]
        locations = [Entity(i, location)
                     for i, location in enumerate(locations)]
        scenes = []
        with open(text_file + '.txt') as infile:
            for start, end in regex_sentence_boundary_gen(infile.read()):
                scenes.append(Scene(start, end))
        for scene in scenes:
            for character in characters:
                for mention, _, _, _ in character.chain:
                    if (entities[mention].start >= scene.start and
                        entities[mention].end <= scene.end):
                        scene.characters.add(character)
            for location in locations:
                for mention, _, _, _ in location.chain:
                    if (entities[mention].start >= scene.start and
                        entities[mention].end <= scene.end):
                        scene.locations.add(location)
        return Story(id, filepath, scenes, set(characters), set(locations))

    def cluster_characters(self):
        """On the basis of co-occurrences of characters in scenes,
        performs a clustering to assign characters to different
        groups."""
        cooccurences = np.zeros((len(self.characters), len(self.characters)))
        for scene in self:
            for character_i in scene.characters:
                for character_j in scene.characters:
                    cooccurences[character_i.id, character_j.id] += 1.0
                    cooccurences[character_j.id, character_i.id] = cooccurences[
                        character_i.id, character_j.id]
        cooccurences = cooccurences / cooccurences.sum()
        clusterer = DBSCAN(eps=cooccurences.mean(), min_samples=1)
        clustering = clusterer.fit_predict(cooccurences)
        for character in self.characters:
            # check if this propagates
            character.cluster = clustering[character.id]

    def _block_scenes(self, window_size=4, smooth=False):
        scenes = []
        for i in range(0, len(self) - window_size, window_size):
            scenes.append(Scene.concat(self[i: i + window_size]))
        self[:] = scenes

    def _storytiling(
        self, dist_between='characters', smoothed=False, window_size=3,
                     window_type='flat', policy='HC', adjacent_gaps=4, k=6):
        gaps = []
        for gap in range(len(self) - 1):
            if gap < (k - 1):
                ws = gap + 1
            elif gap > (len(self) - 1 - k):
                ws = len(self) - 1 - gap
            else:
                ws = k
            lhs = Scene.concat(self[gap - ws + 1: gap + 1])
            rhs = Scene.concat(self[gap + 1: gap + ws + 1])
            gaps.append(1 - lhs.distance(rhs, dist_between=dist_between))
        gaps = np.array(gaps)
        if smoothed:
            gaps = smooth(gaps, window_len=window_size, window=window_type)
        depths = depth_scores(gaps)
        scenes = []
        previous = 0
        for i, boundary in enumerate(identify_boundaries(depths, policy, adjacent_gaps)):
            if boundary == 1:
                scenes.append(Scene.concat(self[previous:i + 1]))
                previous = i + 1
        if previous < len(self):
            scenes.append(Scene.concat(self[previous:]))
        self[:] = scenes

    def _dynamic_clustering(self, t=0.6, criterion='distance', dist_between='characters'):
        dm = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            for j in range(i):
                dm[j, i] = dm[i, j] = self[i].distance(
                    self[j], dist_between=dist_between)
        clusterer = VNClusterer(dm)
        clusterer.cluster()
        Z = clusterer.dendrogram().to_linkage_matrix()
        if criterion == 'distance':
            t = t * max(Z[:,2])
        clusters = fcluster(Z, t, criterion=criterion)
        scenes = []
        for cluster, indexes in groupby(range(len(clusters)), key=lambda i: clusters[i]):
            indexes = list(indexes)
            scenes.append(Scene.concat(self[min(indexes): max(indexes)+1]))
        self[:] = scenes
        return Z

    def scenify(
        self, method='blocks', dist_between='characters', window_size=3, criterion='distance',
                window_type='flat', t=0.6, k=6, policy='HC', smoothed=False, adjacent_gaps=4):
        """Partition the story into a number of scenes. Three methods are supported:
           1) blocks: simple sliding window function;
           2) storytiling: texttiling-like algorithm that tries to find homogeneous blocks of text
              on the basis of the participants involved.
           3) vnc: Variability Neighbor Clustering.
        """
        if method == 'blocks':
            self._block_scenes(window_size=window_size, smooth=smooth)
        elif method == 'storytiling':
            self._storytiling(
                smoothed=smoothed, dist_between=dist_between, window_size=window_size,
                window_type=window_type, policy=policy, adjacent_gaps=adjacent_gaps, k=k)
        elif method == 'vnc':
            return self._dynamic_clustering(t=t, criterion=criterion)
        else:
            raise ValueError("method '%s' is not supported." % method)

    def add_sentiments(self, smoothed=False, window_size=12, binary=False, window_type='flat'):
        """Add a sentiment score to each scene in this story."""
        with codecs.open(self.filepath + '.txt', encoding='utf-8') as infile:
            text = infile.read()
            sentiments = []
            for scene in self:
                sentiments.append(extract_sentiment(
                    Sentence(parse(text[scene.start: scene.end], lemmata=True)))[0])
            if smoothed:
                while window_size > len(self):
                    window_size -= 2
                sentiments = smooth(
                    np.array(sentiments), window_len=window_size, window=window_type)
            for i, sentiment in enumerate(sentiments):
                self[i].sentiment = sentiment if not binary else - \
                    1 if sentiment < 0 else 1

    def to_dict(self, empty_scenes=False):
        """Transform the story into a dictionary."""
        story_dict = {"id": self.id, "scenes":
                      [], "panels": max(scene.end for scene in self)}
        for i, scene in enumerate(self):
            if not empty_scenes and not scene.characters:
                continue
            scene = {"chars": [char.id for char in scene.characters],
                     "locs": [loc.id for loc in scene.locations],
                     "duration": scene.end - scene.start,
                     "start": scene.start,
                     "id": i}
            story_dict["scenes"].append(scene)
        return story_dict

    def to_json(self, empty_scenes=False):
        """Write the story to a json file."""
        with open(self.filepath + '.json', 'w') as out:
            out.write(dumps(self.to_dict(empty_scenes)))

    def to_graph(self):
        """Construct a Graph-like structure from this story."""
        G = nx.Graph()
        G.name = story['id']
        for scene in self:
            for char_i, char_j in combinations(scene.characters, 2):
                G.add_edge(char_i.name, char_j.name)
        return G

    def to_xml(self):
        """Write the characters of this story to an XML file."""
        xml = '<characters>\n'
        xml += '\n'.join(
            '  <character group="%d" id="%d" name="%s" />' % (
                c.cluster, c.id, c.name)
            for c in self.characters)
        xml += '</characters>'
        with open(self.filepath + '.xml', 'w') as out:
            out.write(xml.encode('utf-8'))

    def to_dataframe(self, summed=False, entities='characters', sentiments=False):
        """Transform the story into a n by m matrix where n is the numbers
        of unique characters in the story and m the number of scenes."""
        elements = self.characters if entities == 'characters' else self.locations
        matrix = np.zeros((len(elements), len(self)))
        for i, scene in enumerate(self):
            for entity in getattr(scene,  entities):
                matrix[entity.id, i] = 1 if not sentiments else scene.sentiment
        elements = sorted(elements, key=lambda element: entity.id)
        df = pd.DataFrame(
            matrix, index=[self.id + '-' + entity.name[:-2] for entity in elements])
        if summed:
            df = df.mean(axis=0)
        return df

    def plot(self, kind='signal', summed=False, entities='characters', sentiments=False):
        df = self.to_dataframe(
            summed=summed, entities=entities, sentiments=sentiments)
        if kind == 'signal':
            if summed:
                df.plot()
            else:
                grid_plot(df.values, labels=df.index)

    def sociogram(self, filter=None, binary=True):
        """Construct a sociogram on the basis of the sentiments assigned to the scenes in the story."""
        co_occurrences = defaultdict(lambda: defaultdict(list))
        for scene in self:
            for character_i, character_j in combinations(scene.characters, 2):
                co_occurrences[character_i][
                    character_j].append(scene.sentiment)
        sociogram = nx.Graph()
        for character, neighbors in co_occurrences.items():
            for neighbor, sentiment in neighbors.items():
                sociogram.add_edge(
                    character.name, neighbor.name, weight=mean(sentiment) if not binary else -1 if mean(sentiment) < 0 else 1)
        return Sociogram(sociogram)
