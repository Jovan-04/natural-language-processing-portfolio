class Pretoken:
    def __init__(self, text: str, to_characters: bool = True):
        self.text: str = text
        self.next = None
        if to_characters:
            self.convert_to_characters()


    def convert_to_characters(self):
        """
        Convert the Pretoken instance to one character per node. Called automatically on object creation. 
        """
        a = list(self.text)
        next_node = None
        
        # we want to modify this pretoken so that it's the beginning of our characterized text
        self.text = a[0]
        for char in reversed(a[1:]):
            new_node = Pretoken(char)
            new_node.next = next_node
            next_node = new_node

        self.next = next_node


    def apply_merge_rule(self, rule: str) -> bool:
        """
        Apply a merge rule to the Pretoken. 
        
        :param rule: The merge rule to apply. Two consecutive tokens who, when concatenated, equal `rule` will be combined into one Pretoken node. 
        :type rule: str
        :return: True if a change was made to the Pretoken, False otherwise. 
        :rtype: bool
        """
        current_node = self
        next_node = current_node.next
        made_change = False

        while current_node and next_node:
            if current_node.text + next_node.text == rule:
                current_node.text = current_node.text + next_node.text
                current_node.next = next_node.next
                made_change = True
            
            current_node = current_node.next
            if current_node is not None:
                next_node = current_node.next

        return made_change


    def get_tokens(self) -> list[str]:
        """
        Convert a Pretoken to the tokens making it up. 
        
        :return: The tokens in this Pretoken. 
        :rtype: list[str]
        """
        tokens: list[str] = []

        current = self
        while current.next:
            tokens.append(current.text)
            current = current.next
        
        # the while loop doesn't include the text of the last ('current') token
        # because it breaks from the loop when current.next is None
        tokens.append(current.text)

        return tokens


    # we add equality and hashing based on the string value of the object
    # mostly so we can use a Counter on it
    # this string value should be unique to the data within a Pretoken
    def __eq__(self, value: object) -> bool:
        return self.__str__() == value.__str__()
    
    def __hash__(self) -> int:
        return hash(self.__str__())

    def __str__(self) -> str:
        output = ""
        current_node = self
        while current_node:
            output += current_node.text + " -> "
            current_node = current_node.next

        return output[:-4]
    
    def __repr__(self) -> str:
        return f"'{self.__str__()}'"