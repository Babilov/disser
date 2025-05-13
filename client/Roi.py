class Roi:
    def __init__(self, cords, index):
        self.cords = cords
        self.index = index
        self.car_count = 0
        self.unique_cars_ids = []

    def __len__(self):
        return len(self.cords)
    
    def __str__(self):
        return f"Roi №{self.index} has cords {self.cords} wtih {self.car_count} in."
    
    def __repr__(self):
        return self.__str__()
