function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
 

H = X * Theta';
J = (1/2) * sum(sum(((H-Y).^2).* R));
regularization_factor = ((lambda/2) * sum(sum(Theta.^2))) + ((lambda/2) * sum(sum(X.^2)));
J = J + regularization_factor;


%disp(size(((H-Y).* R)));
%disp(size(Theta));
%disp(size(X));
%disp(R);
%users_that_rated_some_movie = find(R(:,:)==1);
%disp(users_that_rated_some_movie);
%Theta_grad = (((H-Y).* R) * Theta);
%X_grad = (((H-Y).* R) * X);

for i = 1:num_movies
  idx = find(R(i,:)==1);  %index of users that rated movie i 
  Theta_temp = Theta(idx, :);  %parameters of users that evaluated movie i
  Y_temp = Y(i, idx); % ratings for users that evaluated movie i 
  X_grad(i, :) = ((X(i,:) * Theta_temp') - Y_temp) * Theta_temp; 
  regularization_factor = lambda*X(i,:);
  X_grad(i, :) = X_grad(i, :) + regularization_factor;
endfor

for j = 1:num_users
  idx = find(R(:,j)==1);  %index of movies that user j rated 
  X_temp = X(idx, :); % params of movies evaluated by user j
  Y_temp = Y(idx, j); % ratings of all movies evaluated by user j 
  Theta_grad(j, :) = ((Theta(j, :) * X_temp') - Y_temp') * X_temp;  
  regularization_factor = lambda*Theta(j,:);
  Theta_grad(j, :) = Theta_grad(j, :) + regularization_factor;
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
