% Sample calls
cconflevel(2/3, 100000, 1000)
cconflevel(2/5, 100000, 1000)

function [] = cconflevel(a, M, K)
    flag = 1;
    y = zeros(2, M);

    if a > 0.5
        disp('Variance does not exist!!')
    end

    for k = 1:K
        y = y + mcc1d(M, a, flag);
    end
    np = 10;
    Mp = (np:np:M);
    yp = y(:, np:np:end) / K;

    M1 = Mp(1); M2 = Mp(end);
    subplot(2, 1, 1)
    loglog(Mp, yp(2, :), 'b', [M1 M2], yp(2, end) * [(M1/M2)^(-0.5), 1], 'r',...
        [M1 M2], yp(2, end) * [(M1/M2)^(-1/3), 1], 'g')
    legend('|mu-mu_m|', 'm^{-0.5}', 'm^{-1/3}')
    subplot(2, 1, 2)
    plot(Mp, yp(1, :))
    legend('Confidence level')
end

function [y] = mcc1d(M, a, flag)
    U = rand(1, M);
    Y = U .^ (-a);
    mu = mean(Y);
    sigma = std(Y);

    if flag == 1
        q = 1.96; % For CLT bounds at a 95% confidence level
    elseif flag == 2
        q = 2; % For Chebyshev bounds with q = 2
    elseif flag == 3
        q = 1 / a; % For Chebyshev bounds with q = 1 / a
    else
        error('Invalid flag value.');
    end

    bounds = q * sigma / sqrt(M);
    y(1, :) = abs(mu - Y) <= bounds;
    y(2, :) = bounds;
end
