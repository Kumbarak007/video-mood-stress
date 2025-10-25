from tensorflow.keras.models import load_model

def load_emotion_model(model_path):
    """
    Load the emotion classification model and recompile to silence deprecation warnings.
    """
    model = load_model(model_path)
    
    # Recompile with modern loss function
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Matches your training setup
        metrics=['accuracy']
    )
    
    return model