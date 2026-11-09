# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
import argparse
import math
from collections import deque
from typing import List, Optional, Tuple
from apted import APTED, Config
from apted.helpers import Tree
from Levenshtein import distance
from lxml import etree, html
from lxml.html import HtmlElement

class TEDS:

    def __init__(self, structure_only: bool=False, ignore_nodes: Optional[
        str]=None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__: List[str] = []

    def __call__(self, pred: str, gt: str) ->float:
        if not pred or not gt:
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred_element: HtmlElement = html.fromstring(pred, parser=parser)
        gt_element: HtmlElement = html.fromstring(gt, parser=parser)
        xpath_ele = 'body/table'
        if pred_element.xpath(xpath_ele) and gt_element.xpath(xpath_ele):
            pred_element = pred_element.xpath(xpath_ele)[0]
            gt_element = gt_element.xpath(xpath_ele)[0]
            if self.ignore_nodes:
                etree.strip_tags(pred_element, *self.ignore_nodes)
                etree.strip_tags(gt_element, *self.ignore_nodes)
            tree_pred = self.load_html_tree(pred_element)
            tree_true = self.load_html_tree(gt_element)
            distance = APTED(tree_pred, tree_true, CustomConfig()
                ).compute_edit_distance()
            n_nodes_pred = len(pred_element.xpath('.//*'))
            n_nodes_true = len(gt_element.xpath('.//*'))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            return 1.0 - float(distance) / n_nodes
        return 0.0

    def tokenize(self, node: HtmlElement):
        self.__tokens__.append(f'<{node.tag}>')
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append(f'</{node.tag}>')
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node: HtmlElement, parent: Optional[
        HtmlElement]=None) ->Optional[HtmlElement]:
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag, int(node.attrib.get('colspan',
                '1')), int(node.attrib.get('rowspan', '1')), cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node
        return None


class TableTree(Tree):

    def __init__(self, tag: str, colspan: Optional[int]=None, rowspan:
        Optional[int]=None, content: Optional[List[str]]=None, *children:
        Tuple[Tree]):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class CustomConfig(Config):

    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1: TableTree, node2: TableTree) ->float:
        if (node1.tag != node2.tag or node1.colspan != node2.colspan or 
            node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


def Analysis_TED(html1, html2):
    if html1 is None:
        raise ValueError('html1 must be non-empty.')
    if html2 is None:
        raise ValueError('html2 must be non-empty.')
    html1 = wrap_with_table(html1)
    html2 = wrap_with_table(html2)
    teds = TEDS(structure_only=True)
    score = teds(html1, html2)
    return score


def _is_empty_html(s: str) ->bool:
    if s is None:
        return True
    if not isinstance(s, str):
        return True
    return len(s.strip()) == 0


def _normalize_html(s: str) ->str:
    if _is_empty_html(s):
        return '<table><tr><td></td></tr></table>'
    s = s.strip()
    return s


def safe_teds_score(html1: str, html2: str, structure_only: bool=True) ->float:
    if _is_empty_html(html1) and _is_empty_html(html2):
        return 1.0
    if _is_empty_html(html1) != _is_empty_html(html2):
        return 0.0
    html1 = _normalize_html(wrap_with_table(html1))
    html2 = _normalize_html(wrap_with_table(html2))
    teds = TEDS(structure_only=structure_only)
    try:
        score = teds(html1, html2)
        if score is None or isinstance(score, float) and (math.isnan(score) or
            math.isinf(score)):
            return 0.0
        return max(0.0, min(1.0, float(score)))
    except ZeroDivisionError:
        return 1.0 if _is_empty_html(html1) and _is_empty_html(html2) else 0.0
    except Exception:
        return 0.0


def wrap_with_table(html_snippet: str) ->str:
    return f'<html><body><table>{html_snippet}</table></body></html>'


if __name__ == '__main__':
    html3 = '<tr><td rowspan="2">A</td><td></td><td colspan="1">B</td></tr>'
    html4 = '<tr><td rowspan="2">A</td><td colspan="1">B</td></tr>'
    Analysis_TED(html3, html4)
