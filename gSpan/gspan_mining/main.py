"""The main program that runs gSpan."""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import collections
import copy
import itertools
import time
import pandas as pd
import argparse
import networkx as nx
from neo4j import GraphDatabase
from numpy import array
from numpy import argmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder





def str2bool(s):
    """Convert str to bool."""
    return s.lower() not in ['false', 'f', '0', 'none', 'no', 'n']


parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--min_support',
    type=int,
    default=5000,
    help='min support, default 5000'
)
parser.add_argument(
    '-n', '--num_graphs',
    type=float,
    default=float('inf'),
    help='only read the first n graphs in the given database, '
         'default inf, i.e. all graphs'
)
parser.add_argument(
    '-l', '--lower_bound_of_num_vertices',
    type=int,
    default=2,
    help='int, lower bound of number of vertices of output subgraph, default 2'
)
parser.add_argument(
    '-u', '--upper_bound_of_num_vertices',
    type=int,
    default=float('inf'),
    help='int, upper bound of number of vertices of output subgraph, '
         'default inf'
)
parser.add_argument(
    '-d', '--directed',
    type=str2bool,
    default=False,
    help='bool, run for directed graphs, default off, i.e. undirected graphs'
)
parser.add_argument(
    '-v', '--verbose',
    type=str2bool,
    default=False,
    help='bool, verbose output, default off'
)
"""
parser.add_argument(
    'database_file_name',
    type=str,
    help='str, database file name'
)
"""
parser.add_argument(
    '-p', '--plot',
    type=str2bool,
    default=False,
    help='bool, whether plot frequent subgraph, default off'
)
parser.add_argument(
    '-w', '--where',
    type=str2bool,
    default=False,
    help='bool, output where one frequent subgraph appears in database, '
         'default off'
)


output=""
VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1


def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco


class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
        )


class DFScode(list):
    """DFScode is a list of DFSedge."""

    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list()

    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
        )

    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not inplemented yet.')

    def build_rmpath(self):
        """Build right most path."""
        self.rmpath = list()
        old_frm = None
        for i in range(len(self) - 1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):
                self.rmpath.append(i)
                old_frm = frm
        return self

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))


class PDFS(object):
    """PDFS class."""

    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev


class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self


class History(object):
    """History class."""

    def __init__(self, g, pdfs):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
        if pdfs is None:
            return
        while pdfs:
            e = pdfs.edge
            self.edges.append(e)
            (self.vertices_used[e.frm],
                self.vertices_used[e.to],
                self.edges_used[e.eid]) = 1, 1, 1

            pdfs = pdfs.prev
        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history."""
        return self.edges_used[eid] == 1


class gSpan(object):
    """`gSpan` algorithm."""

    def __init__(self,
                 #database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize gSpan instance."""
        #self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        self._frequent_subgraphs = list()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self.timestamps = dict()
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'run']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('Mine:\t{} s'.format(
            time_deltas['run'] - time_deltas['_read_graphs']))
        print('Total:\t{} s'.format(time_deltas['run']))

        return self

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        outputlist=[]
        outputlist = output.split('\n')

        tgraph, graph_cnt = None, 0
        for i, line in enumerate(outputlist):
            cols = line.split(' ')
            
            if cols[0] == 't':
                if tgraph is not None:
                    self.graphs[graph_cnt] = tgraph
                    graph_cnt += 1
                    tgraph = None
                if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                    break
                tgraph = Graph(graph_cnt,
                            is_undirected=self._is_undirected,
                            eid_auto_increment=True)
            elif cols[0] == 'v':
                tgraph.add_vertex(cols[1], cols[2])
            elif cols[0] == 'e':
                tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
            # adapt to input files that do not end with 't # -1'
        if tgraph is not None:
            self.graphs[graph_cnt] = tgraph
        return self

    @record_timestamp
    def _generate_1edge_frequent_subgraphs(self):
        vlb_counter = collections.Counter()
        vevlb_counter = collections.Counter()
        vlb_counted = set()
        vevlb_counted = set()
        for g in self.graphs.values():
            for v in g.vertices.values():
                if (g.gid, v.vlb) not in vlb_counted:
                    vlb_counter[v.vlb] += 1
                vlb_counted.add((g.gid, v.vlb))
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    if self._is_undirected and vlb1 > vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))
        # add frequent vertices.
        for vlb, cnt in vlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                          is_undirected=self._is_undirected)
                g.add_vertex(0, vlb)
                self._frequent_size1_subgraphs.append(g)
                if self._min_num_vertices <= 1:
                    self._report_size1(g, support=cnt)
            else:
                continue
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

    @record_timestamp
    def run(self):
        """Run the gSpan algorithm."""
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        if self._max_num_vertices < 2:
            return
        root = collections.defaultdict(Projected)
        for gid, g in self.graphs.items():
            for vid, v in g.vertices.items():
                edges = self._get_forward_root_edges(g, vid)
                for e in edges:
                    root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                        PDFS(gid, e, None)
                    )

        for vevlb, projected in root.items():
            self._DFScode.append(DFSedge(0, 1, vevlb))
            self._subgraph_mining(projected)
            self._DFScode.pop()

    def _get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _report(self, projected):
        self._frequent_subgraphs.append(copy.copy(self._DFScode))
        if self._DFScode.get_num_vertices() < self._min_num_vertices:
            return
        g = self._DFScode.to_graph(gid=next(self._counter),
                                   is_undirected=self._is_undirected)
        display_str = g.display()
        print('\nSupport: {}'.format(self._support))

        # Add some report info to pandas dataframe "self._report_df".
        self._report_df = self._report_df.append(
            pd.DataFrame(
                {
                    'support': [self._support],
                    'description': [display_str],
                    'num_vert': self._DFScode.get_num_vertices()
                },
                index=[int(repr(self._counter)[6:-1])]
            )
        )
        if self._visualize:
            g.plot()
        if self._where:
            print('where: {}'.format(list(set([p.gid for p in projected]))))
        print('\n-----------------\n')

    def _get_forward_root_edges(self, g, frm):
        result = []
        v_frm = g.vertices[frm]
        for to, e in v_frm.edges.items():
            if (not self._is_undirected) or v_frm.vlb <= g.vertices[to].vlb:
                result.append(e)
        return result

    def _get_backward_edge(self, g, e1, e2, history):
        if self._is_undirected and e1 == e2:
            return None
        for to, e in g.vertices[e2.to].edges.items():
            if history.has_edge(e.eid) or e.to != e1.frm:
                continue
            # if reture here, then self._DFScodep[0] != dfs_code_min[0]
            # should be checked in _is_min(). or:
            if self._is_undirected:
                if e1.elb < e.elb or (
                        e1.elb == e.elb and
                        g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
                    return e
            else:
                if g.vertices[e1.frm].vlb < g.vertices[e2.to] or (
                        g.vertices[e1.frm].vlb == g.vertices[e2.to] and
                        e1.elb <= e.elb):
                    return e
            # if e1.elb < e.elb or (e1.elb == e.elb and
            #     g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
            #     return e
        return None

    def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if min_vlb <= g.vertices[e.to].vlb and (
                    not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history):
        result = []
        to_vlb = g.vertices[rm_edge.to].vlb
        for to, e in g.vertices[rm_edge.frm].edges.items():
            new_to_vlb = g.vertices[to].vlb
            if (rm_edge.to == e.to or
                    min_vlb > new_to_vlb or
                    history.has_vertex(e.to)):
                continue
            if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and
                                       to_vlb <= new_to_vlb):
                result.append(e)
        return result

    def _is_min(self):
        if self._verbose:
            print('is_min: checking {}'.format(self._DFScode))
        if len(self._DFScode) == 1:
            return True
        g = self._DFScode.to_graph(gid=VACANT_GRAPH_ID,
                                   is_undirected=self._is_undirected)
        dfs_code_min = DFScode()
        root = collections.defaultdict(Projected)
        for vid, v in g.vertices.items():
            edges = self._get_forward_root_edges(g, vid)
            for e in edges:
                root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                    PDFS(g.gid, e, None))
        min_vevlb = min(root.keys())
        dfs_code_min.append(DFSedge(0, 1, min_vevlb))
        # No need to check if is min code because of pruning in get_*_edge*.

        def project_is_min(projected):
            dfs_code_min.build_rmpath()
            rmpath = dfs_code_min.rmpath
            min_vlb = dfs_code_min[0].vevlb[0]
            maxtoc = dfs_code_min[rmpath[0]].to

            backward_root = collections.defaultdict(Projected)
            flag, newto = False, 0,
            end = 0 if self._is_undirected else -1
            for i in range(len(rmpath) - 1, end, -1):
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    e = self._get_backward_edge(g,
                                                history.edges[rmpath[i]],
                                                history.edges[rmpath[0]],
                                                history)
                    if e is not None:
                        backward_root[e.elb].append(PDFS(g.gid, e, p))
                        newto = dfs_code_min[rmpath[i]].frm
                        flag = True
            if flag:
                backward_min_elb = min(backward_root.keys())
                dfs_code_min.append(DFSedge(
                    maxtoc, newto,
                    (VACANT_VERTEX_LABEL,
                     backward_min_elb,
                     VACANT_VERTEX_LABEL)
                ))
                idx = len(dfs_code_min) - 1
                if self._DFScode[idx] != dfs_code_min[idx]:
                    return False
                return project_is_min(backward_root[backward_min_elb])

            forward_root = collections.defaultdict(Projected)
            flag, newfrm = False, 0
            for p in projected:
                history = History(g, p)
                edges = self._get_forward_pure_edges(g,
                                                     history.edges[rmpath[0]],
                                                     min_vlb,
                                                     history)
                if len(edges) > 0:
                    flag = True
                    newfrm = maxtoc
                    for e in edges:
                        forward_root[
                            (e.elb, g.vertices[e.to].vlb)
                        ].append(PDFS(g.gid, e, p))
            for rmpath_i in rmpath:
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    edges = self._get_forward_rmpath_edges(g,
                                                           history.edges[
                                                               rmpath_i],
                                                           min_vlb,
                                                           history)
                    if len(edges) > 0:
                        flag = True
                        newfrm = dfs_code_min[rmpath_i].frm
                        for e in edges:
                            forward_root[
                                (e.elb, g.vertices[e.to].vlb)
                            ].append(PDFS(g.gid, e, p))

            if not flag:
                return True

            forward_min_evlb = min(forward_root.keys())
            dfs_code_min.append(DFSedge(
                newfrm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
            )
            idx = len(dfs_code_min) - 1
            if self._DFScode[idx] != dfs_code_min[idx]:
                return False
            return project_is_min(forward_root[forward_min_evlb])

        res = project_is_min(root[min_vevlb])
        return res

    def _subgraph_mining(self, projected):
        self._support = self._get_support(projected)
        if self._support < self._min_support:
            return
        if not self._is_min():
            return
        self._report(projected)

        num_vertices = self._DFScode.get_num_vertices()
        self._DFScode.build_rmpath()
        rmpath = self._DFScode.rmpath
        maxtoc = self._DFScode[rmpath[0]].to
        min_vlb = self._DFScode[0].vevlb[0]

        forward_root = collections.defaultdict(Projected)
        backward_root = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            # backward
            for rmpath_i in rmpath[::-1]:
                e = self._get_backward_edge(g,
                                            history.edges[rmpath_i],
                                            history.edges[rmpath[0]],
                                            history)
                if e is not None:
                    backward_root[
                        (self._DFScode[rmpath_i].frm, e.elb)
                    ].append(PDFS(g.gid, e, p))
            # pure forward
            if num_vertices >= self._max_num_vertices:
                continue
            edges = self._get_forward_pure_edges(g,
                                                 history.edges[rmpath[0]],
                                                 min_vlb,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))
            # rmpath forward
            for rmpath_i in rmpath:
                edges = self._get_forward_rmpath_edges(g,
                                                       history.edges[rmpath_i],
                                                       min_vlb,
                                                       history)
                for e in edges:
                    forward_root[
                        (self._DFScode[rmpath_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].append(PDFS(g.gid, e, p))

        # backward
        for to, elb in backward_root:
            self._DFScode.append(DFSedge(
                maxtoc, to,
                (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
            )
            self._subgraph_mining(backward_root[(to, elb)])
            self._DFScode.pop()
        # forward
        # No need to check if num_vertices >= self._max_num_vertices.
        # Because forward_root has no element.
        for frm, elb, vlb2 in forward_root:
            self._DFScode.append(DFSedge(
                frm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, elb, vlb2))
            )
            self._subgraph_mining(forward_root[(frm, elb, vlb2)])
            self._DFScode.pop()

        return self






class Edge(object):
    """Edge class."""

    def __init__(self,
                 eid=VACANT_EDGE_ID,
                 frm=VACANT_VERTEX_ID,
                 to=VACANT_VERTEX_ID,
                 elb=VACANT_EDGE_LABEL):
        """Initialize Edge instance.

        Args:
            eid: edge id.
            frm: source vertex id.
            to: destination vertex id.
            elb: edge label.
        """
        self.eid = eid
        self.frm = frm
        self.to = to
        self.elb = elb


class Vertex(object):
    """Vertex class."""

    def __init__(self,
                 vid=VACANT_VERTEX_ID,
                 vlb=VACANT_VERTEX_LABEL):
        """Initialize Vertex instance.

        Args:
            vid: id of this vertex.
            vlb: label of this vertex.
        """
        self.vid = vid
        self.vlb = vlb
        self.edges = dict()

    def add_edge(self, eid, frm, to, elb):
        """Add an outgoing edge."""
        self.edges[to] = Edge(eid, frm, to, elb)


class Graph(object):
    """Graph class."""

    def __init__(self,
                 gid=VACANT_GRAPH_ID,
                 is_undirected=True,
                 eid_auto_increment=True):
        """Initialize Graph instance.

        Args:
            gid: id of this graph.
            is_undirected: whether this graph is directed or not.
            eid_auto_increment: whether to increment edge ids automatically.
        """
        self.gid = gid
        self.is_undirected = is_undirected
        self.vertices = dict()
        self.set_of_elb = collections.defaultdict(set)
        self.set_of_vlb = collections.defaultdict(set)
        self.eid_auto_increment = eid_auto_increment
        self.counter = itertools.count()

    def get_num_vertices(self):
        """Return number of vertices in the graph."""
        return len(self.vertices)

    def add_vertex(self, vid, vlb):
        """Add a vertex to the graph."""
        if vid in self.vertices:
            return self
        self.vertices[vid] = Vertex(vid, vlb)
        self.set_of_vlb[vlb].add(vid)
        return self

    def add_edge(self, eid, frm, to, elb):
        """Add an edge to the graph."""
        if (frm is self.vertices and
                to in self.vertices and
                to in self.vertices[frm].edges):
            return self
        if self.eid_auto_increment:
            eid = next(self.counter)
        self.vertices[frm].add_edge(eid, frm, to, elb)
        self.set_of_elb[elb].add((frm, to))
        if self.is_undirected:
            self.vertices[to].add_edge(eid, to, frm, elb)
            self.set_of_elb[elb].add((to, frm))
        return self

    def display(self):
        """Display the graph as text."""
        display_str = ''
        print('t # {}'.format(self.gid))
        for vid in self.vertices:
            print('v {} {}'.format(vid, self.vertices[vid].vlb))
            display_str += 'v {} {} '.format(vid, self.vertices[vid].vlb)
        for frm in self.vertices:
            edges = self.vertices[frm].edges
            for to in edges:
                if self.is_undirected:
                    if frm < to:
                        print('e {} {} {}'.format(frm, to, edges[to].elb))
                        display_str += 'e {} {} {} '.format(
                            frm, to, edges[to].elb)
                else:
                    print('e {} {} {}'.format(frm, to, edges[to].elb))
                    display_str += 'e {} {} {}'.format(frm, to, edges[to].elb)
        return display_str

    def plot(self):
        """Visualize the graph."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception as e:
            print('Can not plot graph: {}'.format(e))
            return
        gnx = nx.Graph() if self.is_undirected else nx.DiGraph()
        vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
        elbs = {}
        for vid, v in self.vertices.items():
            gnx.add_node(vid, label=v.vlb)
        for vid, v in self.vertices.items():
            for to, e in v.edges.items():
                if (not self.is_undirected) or vid < to:
                    gnx.add_edge(vid, to, label=e.elb)
                    elbs[(vid, to)] = e.elb
        fsize = (min(16, 1 * len(self.vertices)),
                 min(16, 1 * len(self.vertices)))
        plt.figure(3, figsize=fsize)
        pos = nx.spectral_layout(gnx)
        nx.draw_networkx(gnx, pos, arrows=True, with_labels=True, labels=vlbs)
        nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
        plt.show()



def main(FLAGS=None):
    """Run gSpan."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "wowhi223"))

    def run_query(input_query):
        # 세션을 열어줍니다.
        with driver.session() as session: 
        # 쿼리를 실행하고 그 결과를 results에 넣어줍니다.
            results = session.run(
                    input_query,
                    parameters={}
                    )
            return results

# 다음과 같이 쿼리하여, 노드, 엣지, 노드를 모두 가져온다.

    input_query = """
    MATCH (n1)-[e]->(n2)
    RETURN n1, e, n2
    """

    results = run_query(input_query)
# result => neo4j.BoltStatementResult object


# 긁어온 쿼리를 다음의 방향성이 있는 그래프에 넣어준다.
    DG = nx.DiGraph()

    for i, path in enumerate(results):
    # 앞서, 쿼리에서 변수명을 n1, n2, e, 로 가져왔으므로 각 값에 할당된 것을 변수에 추가로 넣어준다.
        n1, n2, e = path['n1'], path['n2'], path['e']
    # 그리고, 보통 노드의 경우는 id, labels, properties 로 나누어 정보가 저장되어 있다.
    # 이를 가져오기 편하게, dictionary로 변경한다. 
        n1_dict = {
                'id': path['n1'].id, 
                'labels':path['n1'].labels, 
                'properties':dict(path['n1'])
                }
        n2_dict = {
                'id': path['n2'].id, 
                'labels':path['n2'].labels, 
                'properties':dict(path['n2'])
                }
    # 마찬가지로, edge의 경우도 아래와 같이 정보를 저장한다.
        e_dict = {
                'id':path['e'].id, 
                'type':path['e'].type, 
                'properties':dict(path['e'])
                }

    # 해당 노드를 넣은 적이 없으면 넣는다.
        if n1_dict['id'] not in DG:
            DG.add_nodes_from([
                    (n1_dict['id'], n1_dict)
                    ])
    # 해당 노드를 넣은 적이 없으면 넣는다.
        if n2_dict['id'] not in DG:
            DG.add_nodes_from([
                    (n2_dict['id'], n2_dict)
                    ])
    # edge를 넣어준다. 노드의 경우 중복으로 들어갈 수 있으므로 중복을 체크해서 넣어주지만, 
    # edge의 경우 중복을 체크하지 않아도 문제없다.
        DG.add_edges_from([
                (n1_dict['id'], n2_dict['id'], e_dict)
                ])



    NodeLabel=[]
    NodeLabel.append([])
    NodeLabel.append([])
    EdgeLabel=[]
    FromIdx=[]
    ToIdx=[]
    From=[]
    To=[]
    Name=[]
    NodeLabelEncoded=[]
    EdgeLabelEncoded=[]
    
    for n in DG.nodes(data=True):
        if 'Person' in n[1]['labels']:
            NodeLabel[1].append(n[1]['id'])
            if n[1]['properties']['p_type'] == '개인':
                Name.append(n[1]['properties']['name'])
                NodeLabel[0].append('person')
            else:
                NodeLabel[0].append('person'+ n[1]['properties']['name'])
        if 'Data' in n[1]['labels']:
            Label = 'data'+ n[1]['properties']['name']
            NodeLabel[0].append(Label)        
            NodeLabel[1].append(n[1]['id'])

        if 'Activity' in n[1]['labels']:
            NodeLabel[0].append('activity'+n[1]['properties']['name'])        
            NodeLabel[1].append(n[1]['id'])

		

    NodeLabelArray = array(NodeLabel[0])
    NodeLabel_encoder = LabelEncoder()
    NodeLabelEncoded = NodeLabel_encoder.fit_transform(NodeLabelArray)
    count=0


    NodeLabel1=[]

# 이 전에서 라벨 인코딩 해야 했음
    for name in Name:
    
        del NodeLabel1[:]
        NodeLabel1.append([])
        NodeLabel1.append([])
        del EdgeLabel[:]
        del FromIdx[:]
        del ToIdx[:]
        del From[:]
        del To[:]
        
        n1_dict.clear()
        n2_dict.clear()
        e_dict.clear()
    
        input_query = "match (n1:Person{name:\""+name+"\"})match (n1)-[e1]-(n2)-[e2]-(n3) return n1,n2,n3,e1,e2"
        results = run_query(input_query)
        DG1 = nx.DiGraph()
        
        for i, path in enumerate(results):
        
    # 앞서, 쿼리에서 변수명을 n1, n2, e, 로 가져왔으므로 각 값에 할당된 것을 변수에 추가로 넣어준다.
            n1, n2, n3, e1, e2 = path['n1'], path['n2'], path['n3'], path['e1'], path['e2']
    # 그리고, 보통 노드의 경우는 id, labels, properties 로 나누어 정보가 저장되어 있다.
    # 이를 가져오기 편하게, dictionary로 변경한다. 
        
            n1_dict = {
                    'id': path['n1'].id, 
                    'labels':path['n1'].labels, 
                    'properties':dict(path['n1'])
                    }
            n2_dict = {
                    'id': path['n2'].id, 
                    'labels':path['n2'].labels, 
                    'properties':dict(path['n2'])
                    }
            n3_dict = {
                    'id': path['n3'].id, 
                    'labels':path['n3'].labels, 
                    'properties':dict(path['n3'])
                    }
    # 마찬가지로, edge의 경우도 아래와 같이 정보를 저장한다.
            e1_dict = {
                    'id':path['e1'].id, 
                    'type':path['e1'].type, 
                    'properties':dict(path['e1'])
                    }
            e2_dict = {
                    'id':path['e2'].id, 
                    'type':path['e2'].type, 
                    'properties':dict(path['e2'])
                    }
    # print(e_dict)
    # 해당 노드를 넣은 적이 없으면 넣는다.
            if n1_dict['id'] not in DG1:
                DG1.add_nodes_from([
                        (n1_dict['id'], n1_dict)
                        ])
    # 해당 노드를 넣은 적이 없으면 넣는다.
            if n2_dict['id'] not in DG1:
                DG1.add_nodes_from([
                        (n2_dict['id'], n2_dict)
                        ])
            if n3_dict['id'] not in DG1:
                DG1.add_nodes_from([
                        (n3_dict['id'], n3_dict)
                        ])
    # edge를 넣어준다. 노드의 경우 중복으로 들어갈 수 있으므로 중복을 체크해서 넣어주지만, 
    # edge의 경우 중복을 체크하지 않아도 문제없다.
            DG1.add_edges_from([
                    (n1_dict['id'], n2_dict['id'], e1_dict)
                    ])
            
            DG1.add_edges_from([
                    (n2_dict['id'], n3_dict['id'], e2_dict)
                    ])

    
        for n in DG1.nodes(data=True):
            
            if 'Data' in n[1]['labels']:
            
                Label = 'data'+ n[1]['properties']['name']
                

                NodeLabel1[0].append(NodeLabelEncoded[NodeLabel[0].index(Label)]+2)        
                NodeLabel1[1].append(n[1]['id'])
                
            if 'Activity' in n[1]['labels']:
                Label = 'activity'+n[1]['properties']['name']
                NodeLabel1[0].append(NodeLabelEncoded[NodeLabel[0].index(Label)]+2)        
                NodeLabel1[1].append(n[1]['id'])

            if 'Person' in n[1]['labels']:
                if n[1]['properties']['p_type'] == '개인':
                    Label = 'person'
                else:
                    Label = 'person' + n[1]['properties']['name']
          
                NodeLabel1[0].append(NodeLabelEncoded[NodeLabel[0].index(Label)]+2)
                NodeLabel1[1].append(n[1]['id'])
    
  
        for e in DG1.edges(data=True):
        
            FromIdx.append(NodeLabel1[1].index(e[1]))
            ToIdx.append(NodeLabel1[1].index(e[0]))
            EdgeLabel.append(e[2]['type'])
   


        
        NodeLen = len(NodeLabel1[0])
        EdgeLen = len(EdgeLabel)
    
        EdgeLabelArray = array(EdgeLabel)
        EdgeLabel_encoder = LabelEncoder()    
        EdgeLabelEncoded = EdgeLabel_encoder.fit_transform(EdgeLabelArray)
    

        global output
        output = output+"t # "+str(count)+"\n"
        for i in range(NodeLen):
            output = output + "v " + str(i) + " " + str(NodeLabel1[0][i])+"\n"
        for i in range(EdgeLen):
            output = output + "e " + str(FromIdx[i]) + " " + str(ToIdx[i]) + " " + str(EdgeLabelEncoded[i]+2)+"\n"
        
        count+=1
        DG1.clear()
    output = output + "t # -1"


    if FLAGS is None:
        FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])
    gs = gSpan(
        #database_file_name=FLAGS.database_file_name,
        min_support=FLAGS.min_support,
        min_num_vertices=FLAGS.lower_bound_of_num_vertices,
        max_num_vertices=FLAGS.upper_bound_of_num_vertices,
        max_ngraphs=FLAGS.num_graphs,
        is_undirected=(not FLAGS.directed),
        verbose=FLAGS.verbose,
        visualize=FLAGS.plot,
        where=FLAGS.where
    )

    gs.run()
    gs.time_stats()
    return gs


if __name__ == '__main__':
    main()
    sys.exit(main())