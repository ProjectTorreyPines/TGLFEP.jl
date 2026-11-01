# Ballooning-width physics floor (WIDTH_FLOOR = 0.5): input clamp, solver w_lo clamp, env override.
@testset "ballooning-width physics floor" begin
    @test WIDTH_FLOOR == 0.5
    @test width_floor() == 0.5
    ep = Options{Float64}(1, false, 5, 1, 1, 1)
    ep.WIDTH_IN_FLAG = false; ep.WIDTH_IN = 0.0; ep.WIDTH_MIN = 0.3; ep.WIDTH_MAX = 2.0
    enforce_width_floor!(ep)
    @test ep.WIDTH_IN == 0.0 && ep.WIDTH_MIN == 0.5 && ep.WIDTH_MAX == 2.0   # scan box clamped, placeholder WIDTH_IN untouched
    ep.WIDTH_IN_FLAG = true; ep.WIDTH_IN = 0.2; ep.WIDTH_MIN = 0.3
    enforce_width_floor!(ep)
    @test ep.WIDTH_IN == 0.5 && ep.WIDTH_MIN == 0.3                          # fixed width clamped, unused box untouched
    ep.WIDTH_IN_FLAG = false
    ep.WIDTH_MIN = 1.0; enforce_width_floor!(ep)
    @test ep.WIDTH_MIN == 1.0                        # inputs at/above the floor are untouched
    @test TJLFEP.floor_w_lo(0.05) == 0.5             # the pre-floor solver default is clamped
    @test TJLFEP.floor_w_lo(0.8) == 0.8
    withenv("TJLFEP_WIDTH_FLOOR" => "0.1") do        # numerical-studies override
        @test width_floor() == 0.1
        @test TJLFEP.floor_w_lo(0.2) == 0.2
        ep.WIDTH_MIN = 0.3; enforce_width_floor!(ep); @test ep.WIDTH_MIN == 0.3
    end
end
