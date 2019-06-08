% ====================== DATASET/BASE DE DADOS ======================
% Imgine que você é o CEO do Mc Donaldss e deseja abrir 
% uma nova loja da franquia e precisa decidir qual cidade você escolherá. 
% Para te ajudar nessa decisão, você tem dados correspondentes ao lucro de 
% cada franquia e o tamanho da população da cidade na qual ela se 
% encontra. 

%% ================ Parte I: Carregando os dados ====================

data = load('exdata.txt');
%característica/entrada/feature
X = data(:, 1); 
%saida/alvo/target
y = data(:, 2);

m= length(y); 


%% ===== Definição da Função Custo para Regressão Linear ===
function J = computeCost(X, y, theta)
 
  % Inicializando variáveis
  m = length(y);  
  J = 0;

  %hipótese
  h = X*theta;

  %função custo
  J = sum((h - y).^2)/(2*m);  
 
endfunction

%% ===== Definição do Gradiente Descendente p/  Regressão Linear ===
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
 m = length(y); 
 J_history = zeros(num_iters, 1);
 
 for iter = 1:num_iters
   
   t1 = theta(1) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 1));
   t2 = theta(2) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 2));
   
   theta(1) = t1;
   theta(2) = t2;
   
   %theta -= (alpha/m) * (X' * (X*theta - y));
   J_history(iter) = computeCost(X, y, theta);
   
 endfor
endfunction

%% ================ Parte II: Inicializando os parâmetros do gradiente descendente ====================

% Configurando parâmetros do Gradiente Descendente
X = [ones(m, 1), data(:,1)]; 
theta = zeros(2, 1); 
iterations = 1500;
alpha = 0.01;

% Mostrar o custo inicial
J0 = computeCost(X, y, theta);

fprintf('O custo inicial é %f\n', J0);

%% ================ Parte III: Treinando o gradiente descendente ====================

theta = gradientDescent(X, y, theta, alpha, iterations);

% Exibe os valores dos parâmetros theta1 e theta2 
fprintf('Parâmetros ótimos do modelo: ');
fprintf('%f %f \n', theta(1), theta(2));

%% ================ Parte IV: Testando o gradiente descendente para uma nova amostra ====================

%Predizo lucro da franquia, dado um tamanho da população
predict = [1 7.0] *theta;
fprintf('Para uma população de 70.000 mil habitantes, o lucro predito foi %f\n',...
    predict*10000);




