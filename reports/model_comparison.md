# Transformer IDS - Model Comparison

This report compares the three trained models using the same test dataset.

## Overall Performance

| Model         |   Accuracy |   Macro Precision |   Macro Recall |   Macro F1 |   Weighted Precision |   Weighted Recall |   Weighted F1 |   Training Time (sec) |   Prediction Time (sec) |
|:--------------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|----------------------:|------------------------:|
| Random Forest |     0.9966 |            0.7936 |         0.8423 |     0.8128 |               0.9970 |            0.9966 |        0.9967 |                7.5398 |                  1.3347 |
| LSTM          |     0.8826 |            0.3810 |         0.7610 |     0.4550 |               0.9520 |            0.8826 |        0.9079 |              579.6439 |                  5.8864 |
| Transformer   |     0.9562 |            0.5535 |         0.7784 |     0.6281 |               0.9670 |            0.9562 |        0.9594 |             1434.3816 |                  9.8280 |

## Evaluation Metrics

- Accuracy measures overall classification correctness.
- Macro Precision gives equal importance to each class.
- Macro Recall measures detection capability across classes.
- Macro F1 balances precision and recall across classes.
- Weighted F1 accounts for class frequency.

Because the CICIDS2017 dataset is highly imbalanced, Macro F1 and Macro Recall are particularly important when evaluating minority attack classes.

## Training Time

Training time is reported in seconds and depends on the available CPU hardware and software environment.
