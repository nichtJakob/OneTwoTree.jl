#Chat gpt replacement for StatsBase

function countmap(collection::AbstractVector{T}) where T
    counts = Dict{T, Int}()
    for item in collection
        counts[item] = get(counts, item, 0) + 1
    end
    return counts
end


function mode(collection::AbstractVector)
    counts = countmap(collection)
    # Finde das häufigste Element
    max_count = -1
    mode_element = nothing
    for (key, value) in counts
        if value > max_count
            max_count = value
            mode_element = key
        end
    end

    return mode_element
end

function mean(collection::AbstractVector)
    return sum(collection) / length(collection)
end