import jax 
import jax.numpy as jnp


def construct():
    N = 100
    x_cord = jnp.linspace(0, 1, N)
    y_cord = jnp.linspace(0, 1, N)

    xz, yz = jnp.meshgrid(x_cord, y_cord)

    u = (2 * jnp.pi) * xz
    v = jnp.pi * yz

    def sphere():
        x = jnp.sin(v) * jnp.cos(u)
        y = jnp.sin(v) * jnp.sin(u)
        z = jnp.cos(v)

        shape = (x, y, z)
        return shape

    def cylinder():
        x = jnp.cos(u)
        y = jnp.sin(u)
        z = v

        shape = (x, y, z)
        return shape

    def cone():
        x = v * jnp.cos(u)
        y = v * jnp.sin(u)
        z = v

        shape = (x, y, z)
        return shape

    return sphere(), cylinder(), cone()


def transplant():

    N = 100
    grid = jnp.arange(N**2).reshape(N, N)

    tl = grid[:-1, :-1]
    tr = grid[:-1, 1:]
    bl = grid[1:, :-1]
    br = grid[1:, 1:]

    tri1 = jnp.stack([tl.ravel(), tr.ravel(), bl.ravel()], axis=-1)
    tri2 = jnp.stack([tr.ravel(), br.ravel(), bl.ravel()], axis=-1)

    faces = jnp.concatenate([tri1, tri2], axis=0)
    
    return faces
