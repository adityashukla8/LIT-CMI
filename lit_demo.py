# {{{ imports 

from lit_nlp import dev_server
from lit_nlp import notebook
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.api import dataset as lit_dataset
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.components import gradient_maps, shap_explainer

from typing import Optional

from absl import app
from absl import flags
from absl import logging

from collections.abc import Sequence

import pandas as pd
import numpy as np
import pickle 

from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA

from ipdb import set_trace as ipdb
# }}} 
# {{{ model and data path 

model_file_path = './models/CMI_lgbm.pkl'
CMI_DATA_RESAMPLED_PATH = r'./data/CMI_data_resampled.parquet'
CMI_DATA_VAL_PATH = r'./data/CMI_data_val.parquet'
# }}} 
# {{{ setup 

FLAGS = flags.FLAGS
FLAGS.set_default('default_layout', 'penguins')

modules = layout.LitModuleName
PENGUINE_LAYOUT = layout.LitCanonicalLayout(
    upper={
        'Main': [
            modules.DiveModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    }
)
CUSTOM_LAYOUTS = layout.DEFAULT_LAYOUTS | {'penguins': PENGUINE_LAYOUT}

def threshold_rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_pred(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_rounder(oof_non_rounded, thresholds)
    return -cohen_kappa_score(y_true, rounded_p, weights='quadratic')

class LitDataset(lit_dataset.Dataset):
    def __init__(self, path):
        self.data = pd.read_parquet(path)
        self.feature_columns = [col for col in self.data]
        # self._examples = self.data[self.feature_columns].to_dict(orient='records')
        self._examples = self.data.to_dict(orient='records')
        # print(f"data features: {len(self.feature_columns)}")

    def spec(self):
        feature_spec = {col: lit_types.Scalar() for col in self.feature_columns}
        feature_spec['sii'] = lit_types.CategoryLabel()
        return feature_spec

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, index):
        return self._examples[index]


class LitModel(lit_model.Model):
    def __init__(self, model, n_components=3):
        self.model = model
        self.pca = PCA(n_components=n_components)

    def input_spec(self):
        return {col: lit_types.Scalar() for col in self.model.feature_names_in_}

    def output_spec(self):
        return {
            # "sii": lit_types.CategoryLabel(),
            "sii": lit_types.MulticlassPreds(vocab=[0,1,2,3], parent='sii'),
            "pca_output": lit_types.Embeddings(),
            # "probabilities": lit_types.MulticlassPreds(vocab=[0, 1, 2, 3], null_idx=0, parent='sii', threshold=0.5),  # Add probability outputs
        }

    def predict(self, inputs):
        # print(f"INPUTS TYPE: {type(inputs)}, \n{type(inputs[0]['sii'])}\n\n")
        # print(f"model feature names {len(self.model.feature_names_in_)}: {self.model.feature_names_in_}\n")
        # ipdb()
        inputs_array = np.array([[example[col] for col in self.model.feature_names_in_] for example in inputs])
        # print(f'model input array shape: {inputs_array.shape}')
        # ipdb()
        # val_y = np.array([example['sii'] for example in inputs])
        val_predictions = self.model.predict(inputs_array)
        rounded_predictions = val_predictions.astype(int)
        # predictions_minimized_thresholds = minimize(evaluate_pred, x0=[0.5, 1.5, 2.5], args=(val_y, val_predictions), method='Nelder-Mead')
        # rounded_predictions = threshold_rounder(val_predictions, predictions_minimized_thresholds.x)
        # print(f'PREDICTIONS: {type(predictions)}, {len(predictions)}')

        pca_output = self.pca.fit_transform(inputs_array)

        return [{"sii": [float(pred)], "pca_output": pca_output.tolist()} for pred, pca_output in zip(rounded_predictions, pca_output)]


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    
    FLAGS.set_default('server_type', 'external')
    FLAGS.set_default('demo_mode', True)
    unused = flags.FLAGS(sys.argv, known_only=True)
    
    if unused:
        logging.info('penguin_demo:get_wsgi_app() called with unused args: %s', unused)
    
    return main([])
# }}} 
def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

    lit_model_instance = LitModel(model)
    lit_data_instance = LitDataset(CMI_DATA_VAL_PATH)

    models = {'lgbm model': lit_model_instance}
    data = {'CMI data': lit_data_instance}

    lit_demo = dev_server.Server(
        models=models, 
        datasets=data,
        # interpreters={"gradientmaps": gradient_maps.IntegratedGradients(), "shap": shap_explainer.TabularShapExplainer()},
        # interpreters={"shap": shap_explainer.TabularShapExplainer()},
        layouts=CUSTOM_LAYOUTS, 
        **server_flags.get_flags()
    )
    
    return lit_demo.serve()

if __name__ == '__main__':
    app.run(main)
# lit_demo.serve()
# lit_server = dev_server.Server(models, data, client_root='./lit/lit_nlp/client/build/default/static')
# lit_server.serve()
# lit_widget = notebook.LitWidget(models, data, port=8890)
# lit_widget.render()
