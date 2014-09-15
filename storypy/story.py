import codecs
import operator
import os
import sys

from collections import defaultdict, Counter
from itertools import combinations
from json import dumps

import numpy as np
import pandas as pd
import networkx as nx
import distance

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from pattern.nl import sentiment as extract_sentiment, parse, Sentence

sys.path.append(os.path.expanduser("~") + "/local/brat/server/src")
from annotation import Annotations
from ssplit import regex_sentence_boundary_gen


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
    mentions = graphize(relations, entities)
    for start_entity, _ in mentions.selfloop_edges():  # moet start er nog bij?
        chain = set(entity for edge in nx.bfs_edges(mentions, start_entity)
                    for entity in edge)
        yield chain if chain else set([start_entity])


def read_annotation_file(filename):
    annotations = Annotations(filename, read_only=True)
    entities = {entity.id: entity for entity in annotations.get_entities()
                if entity.type in ('Actor', 'Location')}
    relations = defaultdict(list)
    for relation in annotations.get_relations():
        relations[relation.arg1].append(relation.arg2)
    chains = resolve(relations, entities)
    actors, locations = [], []
    for chain in chains:
        chain = [(entity, entities[entity].tail.strip()) for entity in chain]
        entity_type = entities[chain[0][0]].type
        if entity_type == 'Actor':
            actors.append(chain)
        elif entity_type == 'Location':
            locations.append(chain)
    return actors, locations, entities

# -----------------------------------------------------------------------------
# Classes to create Story objects

class Entity(object):
    """An Enitity represents either a character in a story or a location. 
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
        stopwords = set(w.strip() for w in open('pronouns.txt'))
        try:
            longest_token, _ = Counter([token.lower() for _, token in self.chain
                                        if token.lower() not in stopwords]).most_common()[0]
        except IndexError:
            _, longest_token = max(
                self.chain, key=lambda entity: len(entity[1]))
        longest_token += '-' + str(self.id)
        self.chain = [(id, longest_token) for id, _ in self.chain]
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

    def distance(self, other):
        """Compute the distance between this Scene and some other Scene 
        on the basis of the Jaccard distance between their character sets."""
        if not isinstance(other, Scene):
            raise ValueError("Can't compare to %s" % type(other))
        if not self.characters or not other.characters:
            return 1.0
        return distance.jaccard(self.characters, other.characters)

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
                     self.characters.union(other.characters))

    @staticmethod
    def concat(scenes):
        return reduce(operator.add, scenes)

    def __add__(self, other):
        return self.merge(other)


class Story(list):
    """A Story represents a sequence of Scene objects, where each scene is
    represented by a set of characters and a set of locations."""
    def __init__(self, id, scenes=[], characters=set(), locations=set()):
        assert all(isinstance(scene, Scene) for scene in scenes)
        list.__init__(self, scenes)
        self.id = id
        self.characters = characters
        self.locations = locations

    def distance(self, other):
        if not isinstance(other, Story):
            raise ValueError("Can't compare to %s" % type(other))

    @staticmethod
    def load(filename):
        """Construct a new Story on the basis of some input file."""
        characters, locations, entities = read_annotation_file(
            filename + '.ann')
        characters = [Entity(i, character)
                      for i, character in enumerate(characters)]
        locations = [Entity(i, location)
                     for i, location in enumerate(locations)]
        scenes = []
        with open(filename + '.txt') as infile:
            for start, end in regex_sentence_boundary_gen(infile.read()):
                scenes.append(Scene(start, end))
        for scene in scenes:
            for character in characters:
                for mention, _ in character.chain:
                    if (entities[mention].start >= scene.start and
                        entities[mention].end <= scene.end):
                        scene.characters.add(character)
            for location in locations:
                for mention, _ in location.chain:
                    if (entities[mention].start >= scene.start and
                        entities[mention].end <= scene.end):
                        scene.locations.add(location)
        return Story(filename, scenes, set(characters), set(locations))

    def cluster_characters(self):
        """On the basis of co-occurrences of characters in scenes,
        performs a clustering to assign characters to different
        groups."""
        cooccurences = np.zeros((len(self.characters), len(self.characters)))
        for scene in self:
            for character_i in scene.characters:
                for character_j in scene.characters:
                    cooccurences[character_i, character_j] += 1.0
                    cooccurences[character_j, character_i] = cooccurences[
                        character_i, character_j]
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

    def _character_boundaries(self, smooth=False, window_size=3,
                              policy='HC', adjacent_gaps=4, k=6):
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
            gaps.append(1 - lhs.distance(rhs))
        gaps = np.array(gaps)
        if smooth:
            gaps = smooth(gaps, window_len=window_size, window=window_type)
        depths = _depth_scores(gaps)
        scenes = []
        previous = 0
        for i, boundary in enumerate(_identify_boundaries(depths, policy, adjacent_gaps)):
            if boundary == 1:
                scenes.append(Scene.concat(self[previous:i + 1]))
                previous = i + 1
        if previous < len(self):
            scenes.append(Scene.concat(self[previous:]))
        self[:] = scenes

    def scenify(self, method='blocks', window_size=1, k=6, policy='HC', adjacent_gaps=4):
        """Partition the story into a number of scenes. Two methods are supported: 
           1) simple sliding window function;
           2) texttiling-like algorithm that tries to find homogeneous blocks of text
              on the basis of the participants involved."""
        if method == 'blocks':
            self._block_scenes(window_size=window_size, smooth=smooth)
        elif method == 'char_boundary':
            self._character_boundaries(smooth=smooth, window_size=window_size,
                                       policy=policy, adjacent_gaps=adjacent_gaps, k=k)

    def add_sentiments(self, smooth=False, window_size=12, window_type='flat'):
        """Add a sentiment score to each scene in this story."""
        with codecs.open(self.id + '.txt', encoding='utf-8') as infile:
            text = infile.read()
            sentiments = []
            for scene in self:
                sentiments.append(extract_sentiment(
                    Sentence(parse(text[scene.start: scene.end], lemmata=True)))[0])
            if smooth:
                while window_size > len(self):
                    window_size -= 2
                sentiments = smooth(
                    np.array(sentiments), window_len=window_size, window=window_type)
            for i, sentiment in enumerate(sentiments):
                self[i].sentiment = sentiment

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
        with open(self.id + '.json', 'w') as out:
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
        xml = '\n'.join(
            '  <%s group="%d" id="%d" name="%d" />' % (c.cluster, c.id, c.name)
            for c in self.characters)
        with open(self.id + '.xml', 'w') as out:
            out.write(xml.encode('utf-8'))

    def to_dataframe(self, summed=False, sentiments=False):
        """Transform the story into a n by m matrix where n is the numbers
        of unique characters in the story and m the number of scenes."""
        matrix = np.zeros((len(self.characters), len(self)))
        for i, scene in enumerate(self):
            for character in scene.characters:
                matrix[character, i] = 1 if not sentiments else scene.sentiment
        characters = sorted(self.characters, key=lambda c: c.id)
        df = pd.DataFrame(
            matrix, index=[self.id + '-' + character.name[:-2] for character in characters])
        if summed:
            df = df.mean(axis=0)
        return df
