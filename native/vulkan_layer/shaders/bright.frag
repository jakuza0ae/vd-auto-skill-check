#version 450
// Shadow lift on luminance (keeps colour) + local contrast boosted in dark areas.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform Params {
    vec2 texel;      // 1 / extent
    float strength;  // >= 1, luminance gamma
    float lift;      // black floor
    float detail;    // local contrast amount
} p;

const vec3 W = vec3(0.2126, 0.7152, 0.0722);

void main() {
    vec3 c = texture(src, uv).rgb;
    float l = dot(c, W);

    // Local mean luminance: 8 bilinear taps on a ring (each tap averages 2x2 texels).
    vec2 r = p.texel * 6.0;
    float mean = 0.0;
    mean += dot(texture(src, uv + vec2( r.x, 0.0)).rgb, W);
    mean += dot(texture(src, uv + vec2(-r.x, 0.0)).rgb, W);
    mean += dot(texture(src, uv + vec2(0.0,  r.y)).rgb, W);
    mean += dot(texture(src, uv + vec2(0.0, -r.y)).rgb, W);
    mean += dot(texture(src, uv + r * 0.7071).rgb, W);
    mean += dot(texture(src, uv - r * 0.7071).rgb, W);
    mean += dot(texture(src, uv + vec2(r.x, -r.y) * 0.7071).rgb, W);
    mean += dot(texture(src, uv + vec2(-r.x, r.y) * 0.7071).rgb, W);
    mean *= 0.125;

    float t = p.lift + (1.0 - p.lift) * pow(max(l, 0.0), 1.0 / p.strength);
    t += p.detail * (l - mean) * (1.0 - t);   // more detail where it is still dark
    t = clamp(t, 0.0, 1.0);

    // Scale colour to the new luminance; near black fall back to grey (avoids noise blow-up).
    vec3 o = mix(vec3(t), c * (t / max(l, 1e-4)), smoothstep(0.0, 0.06, l));
    float m = max(o.r, max(o.g, o.b));
    if (m > 1.0) {               // keep hue, then refill lost luminance towards white
        o /= m;
        o += (t - dot(o, W)) * (1.0 - o);
    }
    out_color = vec4(clamp(o, 0.0, 1.0), 1.0);
}
