import json 
from typing import Dict

def pprint_one_level_dict(*dictCollection: Dict) -> None: 
    """
    Allows mutiple arguments, 
    Takes a dictionary and print its prettily by one Level
    
    """

    for x in dictCollection: 
        print("{ ")
        for key, value in x.items(): 
            print(f"    {key}: {value}")
        print("}")
