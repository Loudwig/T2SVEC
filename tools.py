import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

class ToyDataset:
    """
    Conteneur intelligent pour les données de séries temporelles.
    Gère :
    - Opérations arithmétiques (+, -, *, /) entre Datasets OU scalaires.
    - Concaténation.
    - Gestion automatique des IDs de classes (labels).
    """
    def __init__(self, data, labels, timestamps=None, masks=None):
        self.data = data
        self.labels = labels
        self.timestamps = timestamps if timestamps is not None else np.arange(len(data))
        self.masks = masks if masks is not None else {}

    def __len__(self):
        return len(self.data)

    def _get_next_label(self, other=None):
        """Calcule le prochain ID de classe libre."""
        max_self = np.max(self.labels) if len(self.labels) > 0 else 0
        
        max_other = 0
        if isinstance(other, ToyDataset) and len(other.labels) > 0:
            max_other = np.max(other.labels)
            
        return max(max_self, max_other) + 1

    def _create_result(self, new_data, other):
        """Helper pour emballer le résultat dans un nouveau ToyDataset."""
        # Création d'une nouvelle étiquette pour le résultat
        new_label_id = self._get_next_label(other)
        new_labels = np.full(len(new_data), new_label_id, dtype=int)
        
        return ToyDataset(new_data, new_labels, self.timestamps, self.masks)

    # --- ADDITION (+) ---
    def __add__(self, other):
        if isinstance(other, ToyDataset):
            if len(self) != len(other): raise ValueError("Shape mismatch")
            return self._create_result(self.data + other.data, other)
        elif isinstance(other, (int, float)):
            return self._create_result(self.data + other, None)
        return NotImplemented

    def __radd__(self, other): # Cas : 5 + ds
        return self.__add__(other)

    # --- SOUSTRACTION (-) ---
    def __sub__(self, other):
        if isinstance(other, ToyDataset):
            if len(self) != len(other): raise ValueError("Shape mismatch")
            return self._create_result(self.data - other.data, other)
        elif isinstance(other, (int, float)):
            return self._create_result(self.data - other, None)
        return NotImplemented

    def __rsub__(self, other): # Cas : 5 - ds
        # Attention l'ordre compte : other - self
        if isinstance(other, (int, float)):
            return self._create_result(other - self.data, None)
        return NotImplemented

    # --- MULTIPLICATION (*) ---
    def __mul__(self, other):
        if isinstance(other, ToyDataset):
            if len(self) != len(other): raise ValueError("Shape mismatch")
            return self._create_result(self.data * other.data, other)
        elif isinstance(other, (int, float)):
            return self._create_result(self.data * other, None)
        return NotImplemented

    def __rmul__(self, other): # Cas : 5 * ds
        return self.__mul__(other)

    # --- DIVISION (/) ---
    def __truediv__(self, other):
        epsilon = 1e-8
        if isinstance(other, ToyDataset):
            if len(self) != len(other): raise ValueError("Shape mismatch")
            return self._create_result(self.data / (other.data + epsilon), other)
        elif isinstance(other, (int, float)):
            return self._create_result(self.data / (other + epsilon), None)
        return NotImplemented

    def __rtruediv__(self, other): # Cas : 5 / ds
        epsilon = 1e-8
        if isinstance(other, (int, float)):
            return self._create_result(other / (self.data + epsilon), None)
        return NotImplemented
    
    def __getitem__(self, key):
        """
        Permet le slicing et l'indexing : ds[0], ds[10:20], ds[[1, 3, 5]]
        Renvoie toujours un nouvel objet ToyDataset.
        """
        # Si c'est un index unique, on le met en liste pour préserver la dim (N, Features)
        if isinstance(key, int):
            # Gestion des index négatifs
            if key < 0:
                key += len(self)
            key = [key]

        # Application sur les données principales
        new_data = self.data[key]
        new_labels = self.labels[key]
        new_timestamps = self.timestamps[key]
        
        # Application sur les masques
        new_masks = {}
        for k, v in self.masks.items():
            new_masks[k] = v[key]

        return ToyDataset(new_data, new_labels, new_timestamps, new_masks)

    # --- CONCATENATION ---
    @staticmethod
    def concat(datasets):
        if not datasets: return None
        new_data = np.concatenate([ds.data for ds in datasets], axis=0)
        new_labels = np.concatenate([ds.labels for ds in datasets], axis=0)
        new_timestamps = np.arange(len(new_data))
        
        combined_masks = {}
        all_keys = set().union(*[ds.masks.keys() for ds in datasets])
        for k in all_keys:
            mask_list = []
            for ds in datasets:
                mask_list.append(ds.masks.get(k, np.zeros(ds.data.shape, dtype=bool)))
            combined_masks[k] = np.concatenate(mask_list, axis=0)

        return ToyDataset(new_data, new_labels, new_timestamps, combined_masks)

    def copy(self):
        return copy.deepcopy(self)
    def add_noise(self, noise_std=0.1, random_seed=None):
        """Ajoute du bruit in-place sans changer les labels."""
        rng = np.random.default_rng(random_seed)
        noise = rng.normal(0, noise_std, size=self.data.shape)
        self.data += noise
        return self # Pour le chaining


class ToyGenerator:
    """
    Générateur avec mémoire des classes.
    Chaque appel à generate() crée un dataset avec un nouvel ID de classe unique.
    """
    def __init__(self, random_seed=None):
        self.rng = np.random.default_rng(random_seed)
        self.class_counter = 0 # Compteur interne pour les labels

    def reset_counter(self):
        """Remet le compteur de classes à 0."""
        self.class_counter = 0

    def _get_base_signal(self, length, signal_type, **kwargs):
        t = np.arange(length)
        if signal_type == 'sine':
            return kwargs.get('amp', 1.0) * np.sin(2 * np.pi * kwargs.get('freq', 0.1) * t + kwargs.get('phase', 0.0))
        elif signal_type == 'square':
            from scipy import signal as scipy_signal
            return kwargs.get('amp', 1.0) * scipy_signal.square(2 * np.pi * kwargs.get('freq', 0.05) * t)
        elif signal_type == 'linear':
            return kwargs.get('slope', 0.01) * t + kwargs.get('intercept', 0.0)
        elif signal_type == 'flat':
            return np.full(length, kwargs.get('value', 0.0))
        elif signal_type == 'noise':
            return self.rng.normal(0, kwargs.get('std', 1.0), size=length)
        elif signal_type == 'ar_process':
            ar = np.zeros(length)
            ar[0] = self.rng.normal(0, 0.1)
            phi = kwargs.get('phi', 0.9)
            for i in range(1, length):
                ar[i] = phi * ar[i-1] + self.rng.normal(0, 0.1)
            return ar
        else:
            raise ValueError(f"Unknown signal: {signal_type}")

    def generate(self, length=100, type='sine', label_id=None, **kwargs):
        """
        Génère un dataset.
        Si label_id n'est pas fourni, utilise et incrémente le compteur interne.
        """
        data_vec = self._get_base_signal(length, type, **kwargs).reshape(-1, 1)
        
        # Gestion automatique du label ID
        if label_id is None:
            current_id = self.class_counter
            self.class_counter += 1
        else:
            current_id = label_id
            
        labels = np.full(length, current_id, dtype=int)
        
        return ToyDataset(data_vec, labels)

    def generate_composite(self, segments_config, noise_std=0.0):
        """
        Génère une série concaténée.
        Chaque segment aura son propre label unique.
        """
        datasets = []
        for config in segments_config:
            p = config.copy()
            length = p.pop('length', 100)
            s_type = p.pop('type', 'sine')
            # generate() va automatiquement incrémenter le label pour chaque segment
            datasets.append(self.generate(length=length, type=s_type, **p))
            
        full_ds = ToyDataset.concat(datasets)
        
        if noise_std > 0:
            full_ds.data += self.rng.normal(0, noise_std, size=full_ds.data.shape)
            # Note: le bruit additif global ne change pas les classes sous-jacentes ici
            
        return full_ds
    
    # ... (apply_perturbations reste identique) ...
    def apply_perturbations(self, dataset, outlier_ratio=0.0, missing_ratio=0.0, missing_block_len=0):
        ds = dataset.copy()
        T, F = ds.data.shape
        if 'outliers' not in ds.masks: ds.masks['outliers'] = np.zeros((T, F), dtype=bool)
        if 'missing' not in ds.masks: ds.masks['missing'] = np.zeros((T, F), dtype=bool)

        if outlier_ratio > 0:
            n = int(T * outlier_ratio)
            idx = self.rng.choice(T, n, replace=False)
            ds.data[idx, 0] = (np.std(ds.data) * 10) * self.rng.choice([-1, 1], n)
            ds.masks['outliers'][idx, 0] = True
            
        if missing_ratio > 0:
            idx = self.rng.choice(T, int(T * missing_ratio), replace=False)
            ds.data[idx, 0] = 0
            ds.masks['missing'][idx, 0] = True
            
        if missing_block_len > 0:
            start = self.rng.integers(0, T - missing_block_len)
            ds.data[start:start+missing_block_len, 0] = 0
            ds.masks['missing'][start:start+missing_block_len, 0] = True
            
        return ds

def plot_toy_dataset(dataset, title="Toy Dataset", figsize=(15, 5)):
    # ... (La fonction plot précédente était parfaite, pas besoin de la changer)
    # Copiez-collez simplement la fonction plot_toy_dataset de ma réponse précédente ici
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    data = dataset.data.flatten()
    labels = dataset.labels
    timestamps = dataset.timestamps
    masks = dataset.masks
    
    plt.figure(figsize=figsize)
    # Utilisation d'une colormap avec beaucoup de couleurs distinctes
    cmap = plt.get_cmap('tab20') 
    
    start_idx = 0
    current_label = labels[0]
    legend_patches = []
    processed_labels = set()

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            color = cmap(current_label % 20)
            plt.axvspan(timestamps[start_idx], timestamps[i], color=color, alpha=0.15, lw=0)
            if current_label not in processed_labels:
                legend_patches.append(mpatches.Patch(color=color, label=f'Class {current_label}', alpha=0.3))
                processed_labels.add(current_label)
            start_idx = i
            current_label = labels[i]
            
    color = cmap(current_label % 20)
    plt.axvspan(timestamps[start_idx], timestamps[-1], color=color, alpha=0.15, lw=0)
    if current_label not in processed_labels:
        legend_patches.append(mpatches.Patch(color=color, label=f'Class {current_label}', alpha=0.3))

    plt.plot(timestamps, data, color='black', linewidth=1, label='Signal')

    if 'outliers' in masks and np.any(masks['outliers']):
        m = masks['outliers'].flatten()
        plt.scatter(timestamps[m], data[m], color='red', zorder=10, marker='x', s=50, label='Outliers')

    if 'missing' in masks and np.any(masks['missing']):
        m = masks['missing'].flatten()
        plt.plot(timestamps[m], [0]*np.sum(m), 'o', color='gray', markersize=2, label='Missing loc')

    plt.title(title)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_patches,loc='upper left',bbox_to_anchor=(1.05, 1), 
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()

class TimeSeriesSequencer:
    """
    Orchestrateur pour générer des séries longues et complexes compatibles TS2Vec.
    Gère l'assemblage stochastique de motifs (Atomes).
    """
    def __init__(self, generator: ToyGenerator):
        self.gen = generator
        self.rng = generator.rng

    def generate_sequence(self, atom_configs, total_length=1000, 
                          sequencing='random', transition_noise=0.0, 
                          non_stationarity_scale=0.0):
        current_length = 0
        datasets = []
        
        # Préparation des probas
        if sequencing == 'random':
            probs = [c.get('prob', 1.0) for c in atom_configs]
            probs = np.array(probs) / np.sum(probs)
        
        seq_idx = 0
        
        while current_length < total_length:
            # 1. Choix de la config
            if sequencing == 'random':
                config_idx = self.rng.choice(len(atom_configs), p=probs)
            else:
                config_idx = seq_idx % len(atom_configs)
                seq_idx += 1
                
            config = atom_configs[config_idx]
            
            # 2. Durée
            d_mean = config.get('duration_mean', 100)
            d_std = config.get('duration_std', 0)
            segment_len = int(max(10, self.rng.normal(d_mean, d_std)))
            
            if current_length + segment_len > total_length + d_mean: 
                 segment_len = total_length - current_length
                 if segment_len <= 0: break

            # 3. Paramètres
            s_type = config.get('type', 'sine')
            s_params = config.get('params', {}).copy()
            
            # Non-Stationnarité
            offset_shift = 0
            if non_stationarity_scale > 0:
                scale_factor = 1.0 + self.rng.normal(0, non_stationarity_scale)
                offset_shift = self.rng.normal(0, non_stationarity_scale)
                if 'amp' in s_params:
                    s_params['amp'] *= abs(scale_factor)
                
            # --- CORRECTION ICI ---
            # On force le label_id à être l'index de la configuration (0, 1, 2...)
            # Au lieu de laisser le générateur créer un nouvel ID à chaque fois.
            ds_atom = self.gen.generate(length=segment_len, type=s_type, 
                                        label_id=config_idx, **s_params)
            
            if non_stationarity_scale > 0:
                # On ajoute l'offset directement aux données pour ne pas changer les labels via __add__
                ds_atom.data += offset_shift

            datasets.append(ds_atom)
            current_length += segment_len
            
        full_ds = ToyDataset.concat(datasets)
        
        if len(full_ds) > total_length:
            full_ds = full_ds[:total_length]
            
        return full_ds

    def make_ts2vec_input(self, dataset, window_size=None, stride=None, split_ratio=0.8):
        """
        Transforme un ToyDataset en format Tensor pour TS2Vec.
        
        Modifié pour retourner des labels pour CHAQUE timestamp de la fenêtre.
        
        Args:
            dataset (ToyDataset): Le dataset complet.
            window_size (int): Taille de la fenêtre glissante.
            stride (int): Pas de glissement.
            split_ratio (float): Part du train.
            
        Returns:
            X_train: (N_samples, Window_Size, Features)
            y_train: (N_samples, Window_Size)  <-- CHANGÉ (avant c'était N_samples,)
            X_test:  (N_samples, Window_Size, Features)
            y_test:  (N_samples, Window_Size)  <-- CHANGÉ
        """
        data = dataset.data # (T, F)
        labels = dataset.labels # (T,)
        
        # Découpage Train / Test temporel
        split_idx = int(len(data) * split_ratio)
        
        train_raw = data[:split_idx]
        test_raw = data[split_idx:]
        train_lbl_raw = labels[:split_idx]
        test_lbl_raw = labels[split_idx:]

        def sliding_window(arr, win, step):
            # Fonction générique pour découper des fenêtres (marche pour data et labels)
            if len(arr) < win:
                # Gestion des cas où la série est plus petite que la fenêtre
                if arr.ndim == 2:
                    return np.array([]).reshape(0, win, arr.shape[1])
                else:
                    return np.array([]).reshape(0, win)
            
            windows = []
            for i in range(0, len(arr) - win + 1, step):
                windows.append(arr[i : i + win])
                
            return np.stack(windows)

        if window_size is not None:
            step = stride if stride else window_size // 2
            
            # 1. Découpage des données (X)
            X_train = sliding_window(train_raw, window_size, step)
            X_test = sliding_window(test_raw, window_size, step)
            
            # 2. Découpage des labels (y)
            # ICI : On utilise la même logique sliding_window pour avoir (N, Window_Size)
            y_train = sliding_window(train_lbl_raw, window_size, step)
            y_test = sliding_window(test_lbl_raw, window_size, step)
            
        else:
            # Cas où on donne toute la série d'un coup (Batch size = 1)
            X_train = train_raw[np.newaxis, ...]
            X_test = test_raw[np.newaxis, ...]
            
            # On ajoute une dimension pour être cohérent : (1, Longueur_Totale)
            y_train = train_lbl_raw[np.newaxis, ...]
            y_test = test_lbl_raw[np.newaxis, ...]
            
        return X_train, y_train, X_test, y_test