"""Remove tree_flatten's recursive-helper reference cycle without changing its API.

The function object is retained so existing imports in Torch/checkpoint/DeepSpeed
use the same implementation. Its globals remain torch.utils._pytree's globals.
"""


def _tree_flatten_without_cycle(tree, is_leaf=None):
    def helper(node, leaves):
        if tree_is_leaf(node, is_leaf=is_leaf):
            leaves.append(node)
            return _LEAF_SPEC
        node_type = _get_node_type(node)
        children, context = SUPPORTED_NODES[node_type].flatten_fn(node)
        return TreeSpec(node_type, context, [helper(child, leaves) for child in children])
    leaves = []
    try:
        return leaves, helper(tree, leaves)
    finally:
        # Release the self-reference through the closure cell after recursion.
        helper = None


def install_cycle_free_pytree():
    import torch.utils._pytree as pytree
    assert pytree.tree_flatten.__code__.co_freevars == ()
    pytree.tree_flatten.__code__ = _tree_flatten_without_cycle.__code__
