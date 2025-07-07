import core.weightusageanalyzer as wua
import numpy as np
import tensorflow as tf

rng = np.random.default_rng(seed=42)
X = rng.random((1000, 2))
y = (X[:, 0] + X[:, 1] > 1).astype(int)

model = tf.keras.Sequential([
tf.keras.Input(shape=(2,)),
tf.keras.layers.Dense(1, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=70, batch_size=32, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\n✅ Accuracy du modèle : {accuracy:.4f}")
wua.print_flops_report(model, nb_epochs=70, dataset=X)


importance_list = wua.compute_weight_importance(model, X)

for importance, weights, name in importance_list:
    report, norm_importance = wua.generate_report(importance, weights)
    print(f"\n📌 Rapport pour la couche : {name}")
    wua.print_report(report)
    wua.plot_importance_histogram(norm_importance, report["entropy"])

    
print("\n================================\n")

wua.show(model, X)


