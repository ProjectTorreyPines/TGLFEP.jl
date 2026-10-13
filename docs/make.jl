using Documenter, TJLFEP

# Auto-generate the API reference page from the exported symbols collected in
# `TJLFEP.document` (see src/TJLFEP.jl).
open(joinpath(@__DIR__, "src/api.md"), "w") do f
    println(f, "# API Reference\n")
    for page in keys(TJLFEP.document)
        println(f, "## $page\n")
        println(f, "```@docs")
        for item in TJLFEP.document[page]
            println(f, "$item")
        end
        println(f, "```")
    end
end

makedocs(;
    modules=[TJLFEP],
    format=Documenter.HTML(),
    sitename="TGLFEP.jl",
    checkdocs=:none,
    pages=["index.md", "api.md", "License" => "license.md"],
    warnonly=true
)

# Deploy docs
# This function deploys the documentation to the gh-pages branch of the repository.
# The main documentation that will be hosted on
# https://projecttorreypines.github.io/TGLFEP.jl/stable
# will be built from latest release tagged with a version number.
# The development documentation that will be hosted on
# https://projecttorreypines.github.io/TGLFEP.jl/dev
# will be built from the latest commit on the chosen devbranch argument below.
deploydocs(;
    repo="github.com/ProjectTorreyPines/TGLFEP.jl.git",
    target="build",
    branch="gh-pages",
    devbranch="master",
    versions=["stable" => "v^", "v#.#"]
)
