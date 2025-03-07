
class VisionProjectorConfig:
    def __init__(self, mm_projector_type='linear', mm_hidden_size=768, hidden_size=512):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size

class AudioProjectorConfig:
    def __init__(self, mm_projector_type='linear', audio_hidden_size=768, hidden_size=512):
        self.mm_projector_type = mm_projector_type
        self.audio_hidden_size = audio_hidden_size
        self.hidden_size = hidden_size
        
