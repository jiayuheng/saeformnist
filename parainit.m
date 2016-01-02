function theta = parainit(hiddenSize, visibleSize)


W1 = rand(hiddenSize, visibleSize);
W2 = rand(visibleSize, hiddenSize);

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);


theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

