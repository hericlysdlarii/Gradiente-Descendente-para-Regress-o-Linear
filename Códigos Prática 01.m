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

m = length(y); 

X = ([X - min(X)] / [max(X) - min(X)]);

%new_data = zeros(size(data))
%min = min(data)
%max = max(data)

%new_data(:,1) = (data(:,1) - min(:,1))/(max(:,1) - min(:,1))

%X = new_data(:,1)
%y = data(:,2)

scatter(X, y, 'r', 'linewidth', 2);
set(gca, "fontsize", 20);


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

J_history = zeros(num_iters, 2);

for iter = 1:num_iters

t1 = theta(1) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 1));

t2 = theta(2) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 2));

theta(1) = t1;

theta(2) = t2;

J_history(iter,1) = computeCost(X, y, theta);
J_history(iter,2) = iter;

endfor

endfunction

%% ================ Parte II: Inicializando os parâmetros do gradiente descendente ====================

% Configurando parâmetros do Gradiente Descendente
X = [ones(m, 1), X]; 
theta = zeros(2, 1); 
iterations = 1500;

alpha1 = 0.01;
alpha2 = 0.03
alpha3 = 0.1;
alpha4 = 0.3;


% Mostrar o custo inicial
J0 = computeCost(X, y, theta);

fprintf('O custo inicial é %f\n', J0);

%% ================ Parte III: Treinando o gradiente descendente ====================

hold on;

%[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha1, iterations);
%plot(X(:,2), X*theta, 'linewidth', 2);

%[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha2, iterations);
%plot(X(:,2), X*theta, 'linewidth', 2);

[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha3, iterations);
plot(X(:,2), X*theta, 'linewidth', 2);

%[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha4, iterations);
%plot(X(:,2), X*theta, 'linewidth', 2);

title('Predição com Gradiente Descendente p/ RL');
xlabel('Populacao da Cidade', 'fontsize', 20);
ylabel('Lucro', 'fontsize', 20);
legend('Dados', 'Modelo 0.01', 4);

hold off;

figure
hold on;

[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha1, iterations); 
plot(J_history(:,2),J_history(:,1), 'linewidth', 2);

[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha2, iterations);
plot(J_history(:,2),J_history(:,1), 'linewidth', 2);

[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha3, iterations);
plot(J_history(:,2),J_history(:,1), 'linewidth', 2);

[theta, J_history] = gradientDescent(X, y, zeros(2, 1), alpha4, iterations);
plot(J_history(:,2),J_history(:,1), 'linewidth', 2);

xlabel('Iterações', 'fontsize', 20);
ylabel('Custo (J)', 'fontsize', 20);
set(gca, "fontsize", 20);
title('Taxa de Aprendizado');
legend('0.1', '0.01', '0.03', '0.3');

hold off;

%% ================ Parte IV: Testando o gradiente descendente para uma nova amostra ====================

% Exibe os valores dos parâmetros theta1 e theta2 calculados pelo 
%gradiente descendente 
fprintf('Parâmetros ótimos do modelo: ');
fprintf('%f %f \n', theta(1), theta(2));

%Predizo lucro da franquia, dado um tamanho da população
predict = [1 10] *theta;
fprintf('Para uma população de 100.000 mil habitantes, o lucro predito foi %f\n',...
    predict*10000);