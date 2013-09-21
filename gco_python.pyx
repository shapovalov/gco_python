# cython: experimental_cpp_class_def=True

import numpy as np
cimport numpy as np

from libcpp.map cimport map
from libcpp.pair cimport pair

DEF DEBUG_CHECKS = True   # true if laborious parameter checks are needed 

# This is the energy type; should match the EnergyType and
# EnergyTermType in GCOptimization.h
DEF NRG_TYPE_STR = int

IF NRG_TYPE_STR == int:
    ctypedef np.int32_t NRG_DTYPE_t
    ctypedef int NRG_TYPE
    ctypedef map[pair[int,int],int] PW_MAP_T   # map (s1, s2) -> strength
ELSE:
    ctypedef np.float64_t NRG_DTYPE_t
    ctypedef double NRG_TYPE
    ctypedef map[pair[int,int],double] PW_MAP_T   # map (s1, s2) -> strength
    

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        cppclass SmoothCostFunctor:
            NRG_TYPE compute(int s1, int s2, int l1, int l2)
			
        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(NRG_TYPE *)
        void setSmoothCost(NRG_TYPE *)
        NRG_TYPE expansion(int n_iterations)
        NRG_TYPE swap(int n_iterations)
        void setSmoothCostVH(NRG_TYPE* pairwise, NRG_TYPE* V, NRG_TYPE* H)
        void setSmoothCostFunctor(SmoothCostFunctor* f)
        int whatLabel(int node)
        void setLabelCost(int *)
        void setLabel(int node, int label)
        NRG_TYPE compute_energy()

    cdef cppclass GCoptimizationGeneralGraph:
        GCoptimizationGeneralGraph(int n_vertices, int n_labels)
        void setDataCost(NRG_TYPE *)
        void setSmoothCost(NRG_TYPE *)
        void setNeighbors(int, int)
        void setNeighbors(int, int, NRG_TYPE)
        NRG_TYPE expansion(int n_iterations)
        NRG_TYPE swap(int n_iterations)
        void setSmoothCostFunctor(GCoptimizationGridGraph.SmoothCostFunctor* f) # yep, it works
        int whatLabel(int node)
        void setLabelCost(int *)
        void setLabel(int node, int label)
        NRG_TYPE compute_energy()
        
        
cdef cppclass PottsFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    NRG_TYPE strength_
    
    __init__(NRG_TYPE strength):
        this.strength_ = strength
    
    NRG_TYPE compute(int s1, int s2, int l1, int l2):
        return -this.strength_ if l1 == l2 else 0
        
cdef cppclass GeneralizedPottsFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    PW_MAP_T data_
    
    __init__(object data):
        this.data_ = data
    
    NRG_TYPE compute(int s1, int s2, int l1, int l2):
        if l1 != l2: 
            return 0
        else:
            pair = tuple(sorted([s1,s2]))
            return -this.data_[pair]  


def cut_simple(np.ndarray[NRG_DTYPE_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, double, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, double, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCost(<NRG_TYPE*>pairwise_cost.data)
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)


    del gc

    return result, nrg
    
    
def cut_simple_gen_potts(np.ndarray[NRG_DTYPE_t, ndim=3, mode='c'] unary_cost,
        object pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, double, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: dict: (site1, site2) -> strength, where site1 < site2.
        Pixels are ordered by rows of the grid first
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = unary_cost.shape[2]
    
    IF DEBUG_CHECKS:
        cdef np.ndarray[np.int32_t, ndim=2] pix_nums = np.r_[:h*w].reshape(h,w)
        edges = [tuple(sorted(pair)) for pair in zip(pix_nums[:,:-1].flatten(), pix_nums[:,1:].flatten())] + \
                [tuple(sorted(pair)) for pair in zip(pix_nums[:-1,:].flatten(), pix_nums[1:,:].flatten())]
        for edge in edges:
            if edge not in pairwise_cost:
                raise ValueError("Pairwise potential for the edge (%d,%d) is not given" % edge)
            if pairwise_cost[edge] < 0:
                raise ValueError("Pairwise potential for the edge (%d,%d) is negative, "
                                 "which is not allowed in generalized Potts" % edge)

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCostFunctor(<GeneralizedPottsFunctor*>new GeneralizedPottsFunctor(pairwise_cost))
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)


    del gc
    return result, nrg
    

def cut_simple_vh(np.ndarray[NRG_DTYPE_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] costV,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] costH, 
        n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to grid graph.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(width, height, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    costV: ndarray, int32, shape=(width, height)
        Vertical edge weights
    costH: ndarray, int32, shape=(width, height)
        Horizontal edge weights
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    if unary_cost.shape[2] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    cdef int h = unary_cost.shape[1]
    cdef int w = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")
    if costV.shape[0] != w or costH.shape[0] != w or costV.shape[1] != h or costH.shape[1] != h:
        raise ValueError("incorrect costV or costH dimensions.")

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(h, w, n_labels)
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCostVH(<NRG_TYPE*>pairwise_cost.data, <NRG_TYPE*>costV.data, <NRG_TYPE*>costH.data)
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = w
    result_shape[1] = h
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
        
    del gc
    return result, nrg

def energy_of_graph_assignment(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[np.int32_t, ndim=1, mode='c'] assignment) :
    """
    Calculate the energy of a particular assignment of labels to a graph

    Parameters
    ----------
    edges: ndarray, int32, shape(n_edges, 2 or 3)
        Rows correspond to edges in graph, given as vertex indices.
        if edges is n_edges x 3 then third parameter is used as edge weight
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    assigment : ndarray, int32, shape= (n_vertices,)
        Assignments of labels to nodes 
    """
    
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    if unary_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
        
    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)

    for e in edges:
        if len(e) == 3:
            gc.setNeighbors(e[0], e[1], e[2])
        else:
            gc.setNeighbors(e[0], e[1])
                
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCost(<NRG_TYPE*>pairwise_cost.data)

    for i in xrange(n_vertices):
        gc.setLabel(i, assignment[i])


    nrg = gc.compute_energy()

    return nrg

    


def cut_from_graph(np.ndarray[np.int32_t, ndim=2, mode='c'] edges,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] unary_cost,
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] pairwise_cost,
        np.ndarray[np.int32_t, ndim=1, mode='c'] label_cost=None, n_iter=5,
        algorithm='expansion', np.ndarray[NRG_DTYPE_t, ndim=1, mode='c'] weights=None):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    edges: ndarray, int32, shape(n_edges, 2 or 3)
        Rows correspond to edges in graph, given as vertex indices.
        if edges is n_edges x 3 then third parameter is used as edge weight
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: ndarray, int32, shape=(n_labels, n_labels)
        Pairwise potentials for label compatibility
    label_cost: ndarray, int32, shape=(n_labels)
        Label costs       
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """
    if (pairwise_cost != pairwise_cost.T).any():
        raise ValueError("pairwise_cost must be symmetric.")

    if unary_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("unary_cost and pairwise_cost have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, pairwise_cost must be n_labels x n_labels.\n"
            "Got: unary_cost: (%d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1],
                pairwise_cost.shape[0], pairwise_cost.shape[1]))
    if pairwise_cost.shape[1] != pairwise_cost.shape[0]:
        raise ValueError("pairwise_cost must be a square matrix.")
    if label_cost is not None and (label_cost.shape[0] != pairwise_cost.shape[0]):
        raise ValueError("label_cost must be an array of size n_labels.\n")
    if weights is not None and edges.shape[1] == 3:    
        raise ValueError("weights parameter is ambiguous when edges is a 3-column array.")
    if weights is not None and weights.shape[0] != edges.shape[0]:
        raise ValueError("weights vector should contain one weight per edge.")
        
    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = pairwise_cost.shape[0]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for e in edges:
        if e.shape[0] == 3:
            gc.setNeighbors(e[0], e[1], e[2])
        else:
            gc.setNeighbors(e[0], e[1])
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCost(<int*>pairwise_cost.data)
    if label_cost is not None:
        gc.setLabelCost(<int*>label_cost.data)
    
    if weights is None:
        for e in edges:
            if len(e) == 3:
                gc.setNeighbors(e[0], e[1], e[2])
            else:
                gc.setNeighbors(e[0], e[1])
    else:
        for e,w in zip(edges, weights):
            gc.setNeighbors(e[0], e[1], w)
                
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCost(<NRG_TYPE*>pairwise_cost.data)
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)

    del gc
    return result, nrg

    
def cut_from_graph_gen_potts(
        np.ndarray[NRG_DTYPE_t, ndim=2, mode='c'] unary_cost,
        object pairwise_cost, n_iter=5,
        algorithm='expansion'):
    """
    Apply multi-label graphcuts to arbitrary graph given by `edges`.

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(n_vertices, n_labels)
        Unary potentials
    pairwise_cost: dict: (site1, site2) -> strength, where site1 < site2.
        The order of nodes is the same as in unary_cost
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    """

    cdef int n_vertices = unary_cost.shape[0]
    cdef int n_labels = unary_cost.shape[1]

    cdef GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(n_vertices, n_labels)
    for edge, strength in pairwise_cost.items():
        gc.setNeighbors(edge[0], edge[1])
        if edge[0] >= edge[1]:
            raise ValueError("The order of sites in the edge (%d,%d) should be ascending" % edge)
        if strength < 0:
            raise ValueError("Pairwise potential for the edge (%d,%d) is negative, "
                             "which is not allowed in generalized Potts" % edge)
        
    gc.setDataCost(<NRG_TYPE*>unary_cost.data)
    gc.setSmoothCostFunctor(<GeneralizedPottsFunctor*>new GeneralizedPottsFunctor(pairwise_cost))
    cdef NRG_TYPE nrg
    if algorithm == 'swap':
        nrg = gc.swap(n_iter)
    elif algorithm == 'expansion':
        nrg = gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[1]
    result_shape[0] = n_vertices
    cdef np.ndarray[np.int32_t, ndim=1] result = np.PyArray_SimpleNew(1, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(n_vertices):
        result_ptr[i] = gc.whatLabel(i)
        
    del gc
    return result, nrg
