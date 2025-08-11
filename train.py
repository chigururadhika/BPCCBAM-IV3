from model import model

if __name__ == "__main__":

    from data_loader_ import train_ds, test_ds
    RETRAIN_MODEL = False 
    

    if RETRAIN_MODEL:
        EPOCHS = 10
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=EPOCHS
        )
        model.save('trained_model_.keras') # different name then saved weight to avoid deletation of weights
    else:
        model.load_weights('trained_model.keras')
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc*100:.2f}%")


