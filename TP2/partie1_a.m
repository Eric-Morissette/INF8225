% Clear console and workspace
clc;
clear;

% Load the file
load 20news_w100;

% Initialize dimensions
n = 4;
m = size(newsgroups,2);
o = ones(1,m);
i = 1:m;
j = newsgroups;
Y = sparse(i,j,o,m,n);

Theta = rand(4,101)-0.5;
X = documents;
X = [X ; ones(1,16242)];
taux_dapprentissage = 0.0005;

[XA, XV, XT] = create_train_valid_test_splits(X);

converted = false;
while ~converged
	% Calculer le log vraisemblance conditionnelle chaque itération
	w_times_x = Theta * X;
	logSumExp__w_times_x = log(sum(exp(w_times_x)));
	logLikelihood = sum(sum(Y * bsxfun(@minus, w_times_x, logSumExp__w_times_x)));

	% Calculer 	la précision sur l’ensemble d’apprentissage et l’ensemble de validation après chaque itération

	% Calculer la mise à jour pour les paramètres 
end