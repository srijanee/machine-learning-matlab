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
Theta1_gradient = zeros(size(Theta1));
Theta2_gradient = zeros(size(Theta2));

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
yk = zeros(size(y), num_labels);
for i = 1:size(X),	
	yk(i,y(i)) = 1;                 %setting yk = 1 for the label K in the output matrix y 
end

X = [ones(m, 1) X];

for i = 1:m,
	
                                 %FEED FORWARD ALGORITHM to calculate the activation units 
    a1 = X(i,:);                %a1 is the input vector 
	z2 = Theta1 * a1';
	a2 = [1;sigmoid(z2)];       %a2 is the vector of the first hidden units
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);           %a3 is the vector for the ouput unit
    
                                %FINDING THE COST for the all the labels for input X(i,:)
    J = J + ((1/m) * (-yk(i,:) * log(a3) - (1 - yk(i,:)) * log(1 - a3)));
	
                                 %BACKPROPOGATION ALGORITHM to find the gradient
	delta3 = a3 - yk(i,:)';
	delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
	delta2 = delta2(2:end);
	
	Theta1_gradient = Theta1_gradient + delta2*a1;
	Theta2_gradient = Theta2_gradient + delta3*a2';
end

                                   %REGULARIZING 

T1 = Theta1(:,2:end);           %Dropping the weight for the bias units 
T2 = Theta2(:,2:end);
                                
Reg = (lambda/(2*m)) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));

J = J + Reg;

%regularize backpropagated gradients
T1Reg = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
T2Reg = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_gradient = (1/m) * Theta1_gradient + (lambda/m) * T1Reg;
Theta2_gradient = (1/m) * Theta2_gradient + (lambda/m) * T2Reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_gradient(:) ; Theta2_gradient(:)];


end
