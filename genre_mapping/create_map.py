"""Script for semi-automatically creating the super-genres
JSON map from a source token-association JSON"""
from featgraph import scriptutils, jwebgraph, logger
from chromatictools import cli
import more_itertools
import importlib
import json
import sys
import os
from typing import Dict, Union, List


def supergenre_from_tokens(subgenre: str, token_dict: Dict[str,
                                                           Union[str,
                                                                 List[str]]],
                           direct_dict: Dict[str, Union[str, List[str]]],
                           preprocess_list: List[List[str]]) -> List[str]:
  """Get the supergenres of a subgenre from its name's tokenization

  Args:
    subgenre (str): The subgenre name
    token_dict (dict): The dictionary of associations from
      tokens to supergenre(s)
    direct_dict (dict): The dictionary of direct associations
      from subgenre to supergenre(s)
    preprocess_list (list): The list of substring
      substitutions (old-new string pairs)

  Returns:
    list of str: The list of supergenres"""
  if subgenre in direct_dict:
    supers = direct_dict[subgenre]
    if isinstance(supers, str):
      supers = [supers]
    return supers
  tokens = subgenre
  for old, new in preprocess_list:
    tokens = tokens.replace(old, new)
  tokens = tokens.split(" ")
  supers = (token_dict[t] for t in tokens if t in token_dict)
  supers = sorted(
      set(
          more_itertools.flatten(
              [s] if isinstance(s, str) else s for s in supers)))
  return supers


def interactive_root_genres():
  """Interactively ask for root genres"""
  vs = None
  while True:
    vi = input("Root genre (empty to stop): ")
    if vi == "":
      break
    vs = vi if vs is None else ([vs, vi] if isinstance(vs, str) else [*vs, vi])
  return vs


def interactive_update(subgenre: str, token_dict: Dict[str, Union[str,
                                                                  List[str]]],
                       direct_dict: Dict[str, Union[str, List[str]]]) -> bool:
  """Function for interactively updating the source and output structures

  Args:
    subgenre (str): The subgenre name
    token_dict (dict): The dictionary of associations
      from tokens to supergenre(s)
    direct_dict (dict): The dictionary of direct associations
      from subgenre to supergenre(s)

  Returns:
    bool: Whether a skip has been requested or not"""
  s = input("Create association for genre '{}'\n"
            "[T -> token, D -> direct, others skip] ".format(subgenre)).lower()
  if s == "t":
    t = input("Token (empty to skip):      ").strip()
    if t == "":
      return True
    vs = interactive_root_genres()
    if vs is None:
      return True
    token_dict[t] = vs
  elif s == "d":
    vs = interactive_root_genres()
    if vs is None:
      return True
    direct_dict[subgenre] = vs
  else:
    return True
  return False


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Create or update the super-genres
  JSON map from a source token-association JSON"""
  parser = scriptutils.FeatgraphArgParse(description="Run main script")
  parser.add_argument("base_path", help="The base path for the BVGraph files")
  parser.add_argument("--source-tokenized-path",
                      default=os.path.join(os.path.dirname(__file__),
                                           "source_tokenized.json"),
                      help="The path for the map source file")
  parser.add_argument("--source-direct-path",
                      default=os.path.join(os.path.dirname(__file__),
                                           "source_direct.json"),
                      help="The path for the direct associations source file")
  parser.add_argument("--preprocess-path",
                      default=os.path.join(os.path.dirname(__file__),
                                           "preprocess.json"),
                      help="The path for the preprocessing list file")
  parser.add_argument("-O",
                      "--output-path",
                      default=None,
                      help="The path for the output JSON")
  parser.add_argument("-i",
                      "--interactive",
                      action="store_true",
                      help="Interactively fill uncovered subgenres")
  parser.add_argument("--no-update",
                      action="store_true",
                      help="Don't update the source files")
  args = parser.custom_parse(argv)

  # Load rules from JSON
  tokenized_source_json = {}
  direct_source_json = {}
  preprocess_json = []
  if os.path.isfile(args.source_tokenized_path):
    with open(args.source_tokenized_path, "r", encoding="utf-8") as f:
      tokenized_source_json = json.load(f)
  if os.path.isfile(args.source_direct_path):
    with open(args.source_direct_path, "r", encoding="utf-8") as f:
      direct_source_json = json.load(f)
  if os.path.isfile(args.preprocess_path):
    with open(args.preprocess_path, "r", encoding="utf-8") as f:
      preprocess_json = json.load(f)

  # Load graph
  jwebgraph.start_jvm(jvm_path=args.jvm_path)
  importlib.import_module("featgraph.jwebgraph.utils")
  graph = jwebgraph.utils.BVGraph(base_path=args.base_path)

  # Compute output JSON
  output_dict = {}
  tqdm_fn = (lambda x: x) if args.tqdm is None else args.tqdm
  for g in more_itertools.flatten(tqdm_fn(graph.genre())):
    output_dict[g] = supergenre_from_tokens(g, tokenized_source_json,
                                            direct_source_json, preprocess_json)
    while args.interactive and len(output_dict[g]) == 0:
      args.interactive = not interactive_update(g, tokenized_source_json,
                                                direct_source_json)
      output_dict[g] = supergenre_from_tokens(g, tokenized_source_json,
                                              direct_source_json,
                                              preprocess_json)

  # Log coverage
  roots = sorted(set(more_itertools.flatten(output_dict.values())))
  ncovered = sum(len(x) != 0 for x in output_dict.values())
  logger.info("Total roots:       %5s %s", len(roots), roots)
  logger.info("Total subgenres:   %5s", len(output_dict))
  logger.info("Covered subgenres: %5s (%.2f%s)", ncovered,
              ncovered * 100 / len(output_dict), r"%")

  # Save to files
  if args.output_path is not None:
    with open(args.output_path, "w", encoding="utf-8") as f:
      json.dump(output_dict, f, indent=2)
  if not args.no_update:
    with open(args.source_tokenized_path, "w", encoding="utf-8") as f:
      json.dump(tokenized_source_json, f, indent=2)
    with open(args.source_direct_path, "w", encoding="utf-8") as f:
      json.dump(direct_source_json, f, indent=2)
