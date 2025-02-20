import torch

# Paramètres
batch_size, depth, height, width, channels = 2, 4, 5, 5, 3
kernel_size = 3  

# Tensor d'exemple pour X et eta
X = torch.randn(batch_size, depth, height, width, channels)  # X avec la nouvelle taille
eta = torch.randn(kernel_size, kernel_size, kernel_size)  # Eta

# Calculer A : produit scalaire entre les canaux des pixels voisins
A = torch.zeros(batch_size, depth, height, width, kernel_size, kernel_size, kernel_size, kernel_size, kernel_size, kernel_size)  # A avec toutes les dimensions nécessaires

# Boucles sur chaque élément de la sortie
for b in range(batch_size):  # Pour chaque élément du batch
    for t in range(depth):  # Pour chaque élément de profondeur
        for s1 in range(height):  # Pour chaque élément de hauteur
            for s2 in range(width):  # Pour chaque élément de largeur
                # Calculer A pour ce pixel particulier
                for tau1 in range(kernel_size):  # Boucle sur le voisinage du premier pixel
                    for tau2 in range(kernel_size):  # Boucle sur le voisinage du deuxième pixel
                        for sigma11 in range(kernel_size):  # Indices de sigma_11
                            for sigma21 in range(kernel_size):  # Indices de sigma_21
                                for sigma12 in range(kernel_size):  # Indices de sigma_12
                                    for sigma22 in range(kernel_size):  # Indices de sigma_22
                                        # Vérification des indices valides
                                        idx_t1 = t - tau1
                                        idx_s1 = s1 - sigma11
                                        idx_s2 = s2 - sigma21
                                        idx_t2 = t - tau2
                                        idx_s1_2 = s1 - sigma12
                                        idx_s2_2 = s2 - sigma22
                                        
                                        # Vérifier si les indices sont dans les limites
                                        if (0 <= idx_t1 < depth and 0 <= idx_s1 < height and 0 <= idx_s2 < width and
                                            0 <= idx_t2 < depth and 0 <= idx_s1_2 < height and 0 <= idx_s2_2 < width):
                                            # Extraire les valeurs de X pour les indices actuels
                                            X_patch1 = X[b, idx_t1, idx_s1, idx_s2]  # Élément du patch pour le premier pixel voisin
                                            X_patch2 = X[b, idx_t2, idx_s1_2, idx_s2_2]  # Élément du patch pour le second pixel voisin
                                            
                                            # Calculer le produit scalaire entre les canaux
                                            dot_product = torch.sum(X_patch1 * X_patch2)  # Produit scalaire entre les canaux
                                            
                                            # Calculer A pour ce voisinage particulier
                                            A[b, t, s1, s2, tau1, tau2, sigma11, sigma21, sigma12, sigma22] = (dot_product + 1) ** 2

# Sortie de A
print(A.shape)
