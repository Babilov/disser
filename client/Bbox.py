class Bbox:
    def __init__(self, cords, confidence, class_, track_id):
        self.cords = cords
        self.confidence = confidence
        self.class_ = class_
        self.track_id = track_id
    
    def __str__(self):
        return f"cords: {self.cords} with confidence {self.confidence} has class {self.class_}"
    
    def __repr__(self):
        return self.__str__()
