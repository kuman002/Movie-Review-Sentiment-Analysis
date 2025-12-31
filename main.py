from src.utilis import load_review, clean_text
from src.logger import logging
from src.data_preparation import data_prepare
from src.model_creation import create_model


if __name__=="__main__":
    
    # Download and extract the dataset before running this script from this link:
    #https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    train_path = "aclImdb/train"
    test_path = "aclImdb/test"

    train_df = load_review(train_path)
    test_df = load_review(test_path)
    
    train_df['reviews'] = train_df['reviews'].apply(clean_text)
    test_df['reviews'] = test_df['reviews'].apply(clean_text)

    x_train = train_df['reviews'].values
    x_test = test_df['reviews'].values

    y_train = train_df['sentiments'].values
    y_test = test_df['sentiments'].values
    
    logging.info(x_train.shape)
    
    x_train_pad, x_test_pad = data_prepare(x_train, x_test)
    
    model = create_model(x_train_pad, y_train)
    
    model.save("artifacts/imdb_review_model.keras")
    
    loss, accuracy = model.evaluate(x_test_pad, y_test)
    
    logging.info(accuracy)