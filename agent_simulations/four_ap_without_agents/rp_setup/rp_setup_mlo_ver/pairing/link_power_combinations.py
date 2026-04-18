from itertools import combinations, product, chain


def links_power_combs(n_links:int=3, n_power_levels:int=12):
    """
    Generate all possible combinations of links and their corresponding power levels.
w
    Args:
        n_links (int): 
            Number of available links. Default is 3.
        n_power_levels (int):
            Number of power levels per link. Default is 12.

    Returns:
        list: A list of tuples where each tuple contains:
            - link_combination (tuple): Indices of selected links
            - power_tuple (tuple): Power levels assigned to those links
    """

    links = list(range(n_links))
    link_combinations = list(chain.from_iterable(combinations(links, r) for r in range(1, len(links)+1)))

    n_power_levels = 12
    links_power_combination = []

    for lc in link_combinations:
        for p in product(range(n_power_levels), repeat=len(lc)):
            links_power_combination.append((lc, p))

    return links_power_combination


