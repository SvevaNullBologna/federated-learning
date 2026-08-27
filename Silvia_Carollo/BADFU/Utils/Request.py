class Request():
    def __init__(self):
        self.requests = []

    def add(self, id: int, indexes_to_erase: list[int] = None):
        if indexes_to_erase is None:
            print(f"no indexes to erase, the request is useless\n")
            return 
        self.requests.append((id, indexes_to_erase))
    
    def remove(self, id: int):
        self.requests = [req for req in self.requests if req[0] != id]

    def contains(self, id: int) -> bool:
        return any(req[0] == id for req in self.requests)

    def get(self, id: int):
        for req in self.requests:
            if req[0] == id:
                return req 
        return None 