import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, random, lax


def getAccFn(pos, G, softening, chunkSize):
    '''
    This will return a function which can later be used to
    calucate the accelerations for each particles
    '''
    numParticles = pos.shape[0]
    numChunks = int(lax.ceil(numParticles / chunkSize))

    @jit
    def getAcc(pos):
        acc = jnp.zeros_like(pos)
        for chunk in range(numChunks):
            startIdx = int(chunk * chunkSize)
            endIdx = int(min(startIdx + chunkSize, numParticles))

            chunk_pos = pos[startIdx:endIdx]

            dx = chunk_pos[:, 0:1].T - pos[:, 0:1]
            dy = chunk_pos[:, 1:2].T - pos[:, 1:2]
            dz = chunk_pos[:, 2:3].T - pos[:, 2:3]

            inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)**(-1.5)

            ax_chunk = G * jnp.sum(dx * inv_r3, axis=1)
            ay_chunk = G * jnp.sum(dy * inv_r3, axis=1)
            az_chunk = G * jnp.sum(dz * inv_r3, axis=1)
            acc = acc.at[:, startIdx:endIdx].set(
                jnp.column_stack((ax_chunk, ay_chunk, az_chunk)))
        return acc

    return getAcc


def main():
    N = 100         # Number of particles
    t = 0.0         # Start time
    tEnd = 10.0     # End time
    dt = 0.01       # Step size
    softening = 0.1
    '''Softening of particles: Giving the particles themselves basically a
    "contracticable" shell so that force of gravity doesn't explode,
    since the denominator becomes 0 if two particles collide'''
    G = 10.0        # Graviational accelerations
    chunkSize = 128
    '''chunkSize: will be used later when the accelerations get calculated
    since we only have limited amount of vram, and will therefor be used
    to calculate the accelerations in smaller cunks'''
    plotRealTime = True

    key = random.PRNGKey(0)

    arr = [-10, -9, -8, -7, -6, -5, -4, -3, -
           2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    mass = 20.0 * jnp.ones((N, 1)) / N
    pos = random.normal(key, (N, 3), dtype=jnp.float32)
    vel = random.normal(key, (N, 3), dtype=jnp.float32)

    # Convert to Center-of-Mass frame
    vel -= jnp.mean(mass*vel, 0) / jnp.mean(mass)

    accFn = getAccFn(pos, G, softening, chunkSize)

    acc = accFn(pos).block_until_ready()

    Nt = int(jnp.ceil(tEnd/dt))

    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')

    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = accFn(pos).block_until_ready()

        # (1/2) kick
        vel += acc*dt / 2.0

        # update time
        t += dt

        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, color="blue")
            ax1.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
            ax1.set_aspect("equal", "box")
            ax1.set_xticks(arr)
            ax1.set_yticks(arr)
            ax1.set_zticks(arr)
            plt.pause(0.001)

    plt.show()
    return 0


if __name__ == "__main__":
    main()
