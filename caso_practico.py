import pandas as pd
import utils
from decisiontree.demo2 import DecisionTreeStatic
from decisiontree.demo2 import print_tree

# Lectura de dataset
df = pd.read_csv('iris.data.csv')
# Separación de características
X = df.iloc[:, :4]
y = df.iloc[:, -1]
# Conversión de clases a valores numéricos
y = utils.label_encoder(y)
# Separamos en data de entrenamiento y test
X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.33, random_state=42)
X_train = utils.validate_type(X_train)
y_train = utils.validate_type(y_train)
# Creamos árbol
model = DecisionTreeStatic(max_depth=5, min_size=1)
# Entrenamos el árbol y devolvemos la raíz, para recorrerlo posteriormente
root = model.fit(X_train, y_train)
# Calculamos la precisión
print('Score %s' % model.score(X_test, y_test))
# Imprimimos cada nodo y el umbral en cada uno, así como su característica
print_tree(root)

