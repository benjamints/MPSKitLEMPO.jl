struct LinkTransferMatrix <: AbstractTransferMatrix
    F::Union{Function,Missing}
    isflipped::Bool
end

function LinkTransferMatrix(F::Union{Function,Missing})
    return LinkTransferMatrix(F, false)
end

function TensorKit.flip(tm::LinkTransferMatrix)
    return LinkTransferMatrix(tm.F, !tm.isflipped)
end

function (d::LinkTransferMatrix)(vec)
    B = copy(vec)

    if !ismissing(d.F)
        if d.isflipped
            for (t1, t2) in fusiontrees(B[1, end, 1])
                B[end][t1, t2] .+= d.F(t2.uncoupled[1]) .* B[1][t1, t2]
            end
        else
            for (t1, t2) in fusiontrees(B[1, 1, 1])
                B[1][t1, t2] .+= d.F(t1.uncoupled[1]) .* B[end][t1, t2]
            end
        end
    end

    return B
end