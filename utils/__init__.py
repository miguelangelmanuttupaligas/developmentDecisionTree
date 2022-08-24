import numpy
import pandas


def label_encoder(y):
    """Codifica en numero entero las etiquetas"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y)
    print('Clases: ', le.classes_)
    print('Transformación: ', le.transform(le.classes_))
    return le.transform(y)


def validate_type(X):
    if isinstance(X, numpy.ndarray):
        return X.tolist()
    if isinstance(X, pandas.DataFrame) or isinstance(X, pandas.Series):
        return X.values.tolist()
    else:
        return X


def train_test_split(X, y, test_size=0.33, random_state=42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return validate_type(X_train), validate_type(X_test), validate_type(y_train), validate_type(y_test)
