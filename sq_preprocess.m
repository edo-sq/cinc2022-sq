function [res] = sq_preprocess(rec, sz, step)  
    L = floor((length(rec) - sz)/step);
    res = cell(L,1);
    for k = 1:L
        si = L*(k-1)+1;
        ei = min(si + sz, length(rec));
        wr = abs(stft(rec(si:ei), 4000, 'FFTLength', 128));
        res(k) = {wr};
    end
end