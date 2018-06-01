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

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
htx = a3;

pred_1 = log(htx);
pred_0 = log(1 - htx);

%y is an m x 1 matrix, change to m x num_labels matrix
Y = [];
  for val = 1:num_labels
    Y = [Y y==val];
  end

J = -1/m * sum(sum((Y .* pred_1) + ((1 - Y) .* pred_0)));
%add regularization term without the zeros.
J = J + lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)))/(2*m);

% -------------------------------------------------------------
%Time for backprop
  d3 = zeros(num_labels, 1);
  d2 = zeros(hidden_layer_size, 1);
  D1 = zeros(size(Theta1));
  D2 = zeros(size(Theta2));

  for i = 1:m
    %ith example
    %for all labels
    d3 = (a3(i,:) .- Y(i,:))';

    %now to compute d2
    d2 = (Theta2(:, 2:end)' * d3) .* deriv(a2, i)';

    D2 = D2 + d3 * a2(i,:);
    D1 = D1 + d2 * a1(i,:);
  end

  %done with all training examples, now compute final gradients
  %do not count terms that correspond to bias terms for regularization

  Theta1_wo_bias = Theta1;
  Theta1_wo_bias(:,1) = 0;
  Theta2_wo_bias = Theta2;
  Theta2_wo_bias(:,1) = 0;

  Theta1_grad = 1/m * (D1 + lambda * Theta1_wo_bias);
  Theta2_grad = 1/m * (D2 + lambda * Theta2_wo_bias);
% =========================================================================



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function[d] = deriv(a2, i)
  %i is the ith training example
  d = a2(i,2:end) .* (1 - a2(i,2:end));
end
