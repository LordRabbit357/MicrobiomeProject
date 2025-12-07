
x = "ATC"
y = "CAT"
z = "TCA"
indel = -2.5

memo = {(0,0,0): (0, (0,0,0))}
memo_debug = {}

def score(a,b):
    AT = -2
    CA = -2
    TC = -2
    if a == "A" and b == "T":
        return AT
    elif a == "T" and b == "A":
        return AT
    elif a == "C" and b == "A":
        return CA
    elif a == "A" and b == "C":
        return CA
    elif a == "C" and b == "T":
        return TC
    elif a == "T" and b == "C":
        return TC
    else:
        return 0


def s(i,j,k):
    options = []
    if (i,j,k) in memo:
        return memo[(i,j,k)]
    if i > 0 and j > 0 and k > 0:
        t = (s(i-1,j-1,k-1)[0] + score(x[i-1], y[j-1]) + score(x[i-1], z[k-1]) + score(y[j-1], z[k-1]), (i-1,j-1,k-1))
        options.append(t)
    if i > 0:
        t = (s(i-1,j,k)[0] + (2*indel), (i-1,j,k))
        options.append(t)
    if j > 0:
        t = (s(i,j-1,k)[0] + (2*indel), (i,j-1,k))
        options.append(t)
    if k > 0:
        t = (s(i,j,k-1)[0] + (2*indel), (i,j,k-1))
        options.append(t)
    if i > 0 and j > 0:
        t  = (s(i-1, j-1, k)[0] + (2*indel) + score(x[i-1], y[j-1]), (i-1, j-1, k))
        options.append(t)
    if i > 0 and k > 0:
        t = (s(i-1, j, k-1)[0] + (2*indel) + score(x[i-1], z[k-1]), (i-1, j, k-1))
        options.append(t)
    if k > 0 and j > 0:
        t = (s(i, j-1, k-1)[0] + (2*indel) + score(y[j-1], z[k-1]), (i, j-1, k-1))
        options.append(t)

    memo[(i,j,k)] = max(options, key=lambda x: x[0])
    memo_debug[(i,j,k)] = options
    
    return max(options)

s(len(x),len(y),len(z))

test = memo[(len(x),len(y),len(z))]
prev = (3,3,3)
lines = ["", "", ""]
while test != memo[(0,0,0)]:
    print(test, prev)
    for i in range(len(lines)):
        if test[1][i] == prev[i]:
            lines[i] += "-"
        else:
            if i == 0:
                lines[i] += x[test[1][i]]
            if i == 1:
                lines[i] += y[test[1][i]]
            if i == 2:
                lines[i] += z[test[1][i]]
    prev = test[1]
    test = memo[test[1]]

for line in lines:
    print(line[::-1])


