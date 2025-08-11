import pandas as pd
import logging
from datasets import load_dataset
from sklearn.utils import resample
from config import Config

logger = logging.getLogger(__name__)


class DataLoaderModule:
    """Unified data loading interface"""
    
    @staticmethod
    def load_imdb_data():
        """Load IMDB dataset"""
        try:
            dataset = load_dataset("imdb")
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])
            
            # Map labels: 0->0 (negative), 1->2 (positive), skip neutral
            train_df['label'] = train_df['label'].map({0: 0, 1: 2})
            test_df['label'] = test_df['label'].map({0: 0, 1: 2})
            train_df['source'] = 'imdb'
            test_df['source'] = 'imdb'
            
            return train_df, test_df
        except Exception as e:
            logger.error(f"Failed to load IMDB data: {e}")
            raise

    @staticmethod
    def load_sst_data():
        """Load Stanford Sentiment Treebank dataset"""
        try:
            dataset = load_dataset("sst", "default")
            train_df = pd.DataFrame(dataset['train'])
            val_df = pd.DataFrame(dataset['validation'])
            
            # Rename 'sentence' column to 'text'
            if 'sentence' in train_df.columns:
                train_df = train_df.rename(columns={'sentence': 'text'})
            if 'sentence' in val_df.columns:
                val_df = val_df.rename(columns={'sentence': 'text'})
            
            # Map SST labels to 3-class sentiment
            def map_sst_labels(label):
                if label <= 1:
                    return 0  # negative
                elif label == 2:
                    return 1  # neutral
                else:
                    return 2  # positive
            
            train_df['label'] = train_df['label'].map(map_sst_labels)
            val_df['label'] = val_df['label'].map(map_sst_labels)
            train_df['source'] = 'sst'
            val_df['source'] = 'sst'
            
            return train_df, val_df
        except Exception as e:
            logger.error(f"Failed to load SST data: {e}")
            raise

    @staticmethod
    def load_tweeteval_data():
        """Load TweetEval sentiment dataset"""
        try:
            dataset = load_dataset("tweet_eval", "sentiment")
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])
            
            train_df['source'] = 'tweeteval'
            test_df['source'] = 'tweeteval'
            
            return train_df, test_df
        except Exception as e:
            logger.error(f"Failed to load TweetEval data: {e}")
            raise

    @staticmethod
    def clean_text_data(df):
        """Clean text data by handling NaN values and empty strings"""
        if 'text' not in df.columns:
            logger.error(f"'text' column not found. Available columns: {list(df.columns)}")
            raise KeyError(f"'text' column not found. Available columns: {list(df.columns)}")
        
        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip() != '']
        
        return df.reset_index(drop=True)

    @staticmethod
    def create_balanced_dataset(imdb_train, sst_train, tweet_train, samples_per_class=10000):
        """Create balanced dataset from multiple sources"""
        logger.info(f"Creating balanced dataset with {samples_per_class} samples per class...")
        
        # Clean all datasets
        imdb_train = DataLoaderModule.clean_text_data(imdb_train)
        sst_train = DataLoaderModule.clean_text_data(sst_train)
        tweet_train = DataLoaderModule.clean_text_data(tweet_train)
        
        balanced_data = []
        for label in [0, 1, 2]:
            imdb_samples = imdb_train[imdb_train['label'] == label]
            sst_samples = sst_train[sst_train['label'] == label]
            tweet_samples = tweet_train[tweet_train['label'] == label]
            
            all_samples = pd.concat([imdb_samples, sst_samples, tweet_samples], ignore_index=True)
            
            if len(all_samples) >= samples_per_class:
                selected = all_samples.sample(n=samples_per_class, random_state=42)
            else:
                selected = resample(all_samples, n_samples=samples_per_class, random_state=42)
            
            balanced_data.append(selected)
            logger.info(f"Class {label}: {len(selected)} samples")
        
        final_dataset = pd.concat(balanced_data, ignore_index=True)
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return final_dataset

    @staticmethod
    def create_test_set(imdb_test, sst_test, tweet_test, samples_per_class=1500):
        """Create balanced test set from multiple sources"""
        # Clean all test datasets
        imdb_test = DataLoaderModule.clean_text_data(imdb_test)
        sst_test = DataLoaderModule.clean_text_data(sst_test)
        tweet_test = DataLoaderModule.clean_text_data(tweet_test)
        
        test_data = []
        for label in [0, 1, 2]:
            imdb_samples = imdb_test[imdb_test['label'] == label]
            sst_samples = sst_test[sst_test['label'] == label]
            tweet_samples = tweet_test[tweet_test['label'] == label]
            
            all_test = pd.concat([imdb_samples, sst_samples, tweet_samples], ignore_index=True)
            
            if len(all_test) >= samples_per_class:
                selected = all_test.sample(n=samples_per_class, random_state=42)
            else:
                selected = all_test
            
            test_data.append(selected)
        
        final_test = pd.concat(test_data, ignore_index=True)
        final_test = final_test.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return final_test


def load_combined_data(config: Config):
    """Load and combine multiple datasets"""
    try:
        # Load individual datasets
        imdb_train, imdb_test = DataLoaderModule.load_imdb_data()
        sst_train, sst_val = DataLoaderModule.load_sst_data()
        tweet_train, tweet_test = DataLoaderModule.load_tweeteval_data()
        
        # Create balanced datasets
        train_df = DataLoaderModule.create_balanced_dataset(
            imdb_train, sst_train, tweet_train, 
            samples_per_class=config.samples_per_class
        )
        
        test_df = DataLoaderModule.create_test_set(
            imdb_test, sst_val, tweet_test, 
            samples_per_class=config.test_samples_per_class
        )
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Failed to load combined data: {e}")
        raise