import random
import json
from collections import namedtuple

import messenger.envs.config as config

Descr = namedtuple("Description", ['entity', 'role', 'type'])

class TextManual:
    '''
    Class which implements methods that allow environments to construct
    text manuals for games in Messenger.
    '''
    
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.descriptors = json.load(f)

    def get_descriptor(self, entity, role, entity_type, no_type_p=0.15):
        '''
        Get a descriptor using the templates.
        Parameters:
        entity: The object that is being described (e.g. alien)
        role: The role the the object plays in the env. (e.g. enemy)
        entity_type:
            The object type (e.g. chaser)
        no_type_p:
            The probability of returning a descriptor that does not have
            any type information (only if entity_type is not None).
        '''
        if random.random() < no_type_p: # no type information
            return random.choice(self.descriptors[entity][role]["unknown"])

        else:
            return random.choice(self.descriptors[entity][role][entity_type])

    def get_document(self, enemy, message, goal, shuffle=False, 
                enemy_type=None, message_type=None, goal_type=None,
                append=False, delete=False, **kwargs):
        '''
        Makes a document for Messenger using the specified parameters.
        If no type is provided, a random type will be selected.

        Parameters:
        append: 
            If True, append an extraneous sentence to the document describing a 
            random object that is not in {enemy, message, goal}.
        delete: If True, Delete a random descriptor from the document.
        shuffle: 
            If True, shuffles the order of the descriptors
        kwargs:
            All other kwargs go to get_descriptor()
        '''

        document = [
            self.get_descriptor(entity=enemy, entity_type=enemy_type, role="enemy", **kwargs),
            self.get_descriptor(entity=message, entity_type=message_type, role="message", **kwargs),
            self.get_descriptor(entity=goal, entity_type=goal_type, role="goal", **kwargs)
        ]

        if delete: # delete a random descriptor
            document = random.sample(document, 2)

        if append:
            # choose an object not in {enemy, message, goal}
            valid_objs = [obj.name for obj in config.NPCS if obj.name not in [enemy, message, goal]]
            rand_obj = random.choice(valid_objs)
            result = None
            while result is None:
                try:
                    result = self.get_descriptor(
                        entity=rand_obj,
                        role=random.choice(("enemy", "message", "goal")),
                        entity_type=random.choice(("chaser", "fleeing", "immovable")),
                        **kwargs
                    )
                except:
                    pass
            document.append(result)
            
        if shuffle:
            document = random.sample(document, len(document))

        return document

    def get_document_plus(self, *args, **kwargs):
        '''
        Makes a document for Messenger using the specified parameters.

        Parameters:
        args: List of Descrip namedtuples
        '''

        document = []

        for descrip in args:
            document.append(
                self.get_descriptor(
                    entity=descrip.entity,
                    role=descrip.role,
                    entity_type=descrip.type,
                    no_type_p=0,
                    **kwargs
                )
            )

        return document

    def get_decoy_descriptor(self, entity, not_of_role, not_of_type, **kwargs):
        '''
        Get a description about the entity where the entity is not of role not_of_role
        and not of type not_of_type
        '''
        possible_roles = [x for x in ('message', 'goal', 'enemy') if x != not_of_role]
        random.shuffle(possible_roles)
        selected_type = random.choice([x for x in ('chaser', 'fleeing', 'immovable') if x != not_of_type])

        for role in possible_roles:
            try:
                return self.get_descriptor(
                    entity=entity,
                    role=role,
                    entity_type=selected_type,
                    no_type_p=0,
                    **kwargs
                )
            except:
                continue
        raise Exception('decoy description with impossible constraints')


if __name__ == "__main__":
    # just some quick and dirty tests
    from pathlib import Path
    text_json = Path(__file__).parent.joinpath('texts', 'text_train.json')
    manual = TextManual(json_path=text_json)
    descriptions = (
        Descr(entity="airplane", role='goal', type='chaser'),
        Descr(entity="airplane", role='message', type='fleeing'),
        Descr(entity="dog", role='enemy', type='immovable'),
        Descr(entity="dog", role='goal', type='fleeing'),
        Descr(entity="mage", role='goal', type='chaser'),
        Descr(entity="mage", role='message', type='immovable'),
    )
    print(manual.get_document_plus(*descriptions))