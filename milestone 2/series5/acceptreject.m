%Function generates n samples of a discrete random variable with pdf
% 1    2    3    4    5    6    7    8    9    10
%0.11 0.12 0.09 0.08 0.12 0.10 0.09 0.09 0.10 0.10 
%using the acceptance-rejection method. A histogram of the realizations is
%generated.
% Input:  n    number of samples
acceptreject1(1000)
function[] =acceptreject1(n)
    disp('Discrete Acceptance Rejection');
    x = ar_randy(n);
    hist(x);


    function x = ar_randy(n)
        p = [0.11 0.12 0.09 0.08 0.12 0.10 0.09 0.09 0.10 0.10];
        j=0;
        x = zeros(n,1);
        while j<n
            y = discreterandu(1);
            u = rand(1,1);
            c = 1.2;
            if u <= p(y) / (c * 0.10) 
               j=j+1;
               x(j,1)=y;
            end
        end
    end
    
    function x = discreterandu(n)
       x = ceil(10 .* rand(n,1));
    end
end