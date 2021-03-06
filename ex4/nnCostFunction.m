function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1) X];
unit = eye(num_labels);
y = unit(y, :);

% Theta1 = (hidden, input + 1)
% Theta2 = (output, hidden + 1)

a1 = X; % (m, input + 1)
z2 = a1 * Theta1'; % (m, input + 1) * (input + 1, hidden) = (m, hidden)
a2 = sigmoid(z2); % (m, hidden)
a2 = [ones(m,1) a2]; % (m, hidden + 1)
z3 = a2 * Theta2'; % (m, hidden + 1) * (hidden + 1, output) = (m, output)
a3 = sigmoid(z3); % (m, output)
delta3 = a3 - y; % (m, output)
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z2); % (m, output) * (output, hidden) = (m, hidden)
Theta1_grad = delta2' * a1; % (hidden, m) * (m, input + 1) = (hidden, input + 1)
Theta2_grad = delta3' * a2; % (output, m) * (m, hidden + 1) = (output, hidden + 1)
J = sum(sum( -y .* log(a3) -(1-y) .* log(1-a3) ));

%{
for t = 1:m
	a1 = X(t,:)';
	a1 = [1;a1];
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	y_vec = (y(t) == [1:10]')(1:size(a3,1));
	delta3 = a3 - y_vec;
	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);
	Theta1_grad += delta2 * a1';
	Theta2_grad += delta3 * a2';
	J += sum( -y_vec .* log(a3) -(1-y_vec) .* log(1-a3) );
end
%}

Theta1_sum = sum(sum(Theta1(:,2:end).^2));
Theta2_sum = sum(sum(Theta2(:,2:end).^2));
J = J/m + (lambda/(2*m)) * ( Theta1_sum + Theta2_sum );

Regular = (lambda/m) * [zeros(hidden_layer_size,1) Theta1(:,2:end)]; % (hidden, input + 1)
Theta1_grad = Theta1_grad/m + Regular;
Regular = (lambda/m) * [zeros(num_labels,1) Theta2(:,2:end)]; % (output, hidden + 1)
Theta2_grad = Theta2_grad/m + Regular;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
