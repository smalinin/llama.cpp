#include "testing.h"

#include "mtmd-image.h"
#include "mtmd-internal.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// this test file contains:
// 1. test cases for mtmd helpers
// 2. test cases for internal mtmd components
// internal headers can be included here

struct test_registry {
    using fn_t = void (*)(testing &);

    struct entry {
        std::string name;
        fn_t fn;
    };

    static std::vector<entry> & all() {
        static std::vector<entry> entries;
        return entries;
    }

    test_registry(const char * name, fn_t fn) {
        all().push_back({ name, fn });
    }
};

#define MAKE_TEST(name)                                               \
    static void name(testing & t);                                    \
    static const test_registry test_registry_ ## name(#name, &name);  \
    static void name(testing & t)


//
// mtmd_image
//

MAKE_TEST(test_image_preprocessor_lfm2) {
    clip_hparams hparams;
    hparams.patch_size = 16;
    hparams.n_merge = 2;
    hparams.set_limit_image_tokens(64, 256);

    // { image size, expected tiling }
    const std::vector<std::pair<clip_image_size, bool>> cases = {
        { {  704, 704 }, false },
        // 720 / (patch_size * n_merge) is exactly 22.5, so this only matches HF
        // if round_by_factor rounds half to even (22) instead of away from zero (23)
        { {  720, 720 }, false },
        { {  736, 736 }, true  },
        { { 1024, 977 }, true  },
        { { 1056, 384 }, false },
    };

    for (const auto & [size, expected] : cases) {
        const bool actual = mtmd_image_preprocessor_lfm2::should_tile(hparams, size);

        t.assert_equal(
            "tiling for " + std::to_string(size.width) + "x" + std::to_string(size.height),
            std::string(expected ? "tiled" : "single"),
            std::string(actual   ? "tiled" : "single"));
    }
}

MAKE_TEST(test_image_preprocessor_deepseek41v) {
    struct test_case {
        int width;
        int height;
        int n_llm_h;
        int n_llm_w;
        int best_height;
        int best_width;
    };

    const std::vector<test_case> cases = {
        {  1024,  1024,  25,   25,  1036,  1036 },
        {  4000,   500,  11,   88,   462,  3696 },
        {   500,  4000,  80,   10,  3360,   420 },
        {   320,   200,  11,   17,   434,   700 },
        {  1920,  1080,  23,   41,   966,  1708 },
        {     1, 10000, 511,    1, 21462,    42 },
        { 10000,     1,   1, 1021,    42, 42882 },
    };

    for (const auto & expected : cases) {
        const auto actual = mtmd_image_preprocessor_deepseek41v::plan_image_grid(
            expected.width, expected.height, 14, 3, 1024, 295936, 0);
        const std::string size = std::to_string(expected.width) + "x" + std::to_string(expected.height);
        t.assert_equal(size + " n_llm_h", expected.n_llm_h, actual.n_llm_h);
        t.assert_equal(size + " n_llm_w", expected.n_llm_w, actual.n_llm_w);
        t.assert_equal(size + " height", expected.best_height, actual.best_height);
        t.assert_equal(size + " width", expected.best_width, actual.best_width);
        t.assert_true(size + " token budget", actual.n_tokens() <= 1024);
    }
}

MAKE_TEST(test_deepseek41v_layout) {
    const std::vector<int32_t> expected = { 6, 0, 1, 2, 8, 3, 4, 5, 8, 7 };
    const auto actual = dsv41_build_layout_indices(3, 2);
    t.assert_true("row-major layout", expected == actual);
    t.assert_equal("V4.1 output tokens", 10, dsv41_n_output_tokens(3, 2));
    t.assert_equal("V4 layout unchanged", 12, dsv4_get_block_layout(3, 2, 2).n_out);
}

MAKE_TEST(test_deepseek41v_media_separator) {
    t.assert_equal("missing trailing separator", 2, mtmd_dsv41_separator_padding("text", false));
    t.assert_equal("one trailing newline",       1, mtmd_dsv41_separator_padding("text\n", false));
    t.assert_equal("complete trailing separator", 0, mtmd_dsv41_separator_padding("text\n\n", false));
    t.assert_equal("missing leading separator",  2, mtmd_dsv41_separator_padding("text", true));
    t.assert_equal("one leading newline",        1, mtmd_dsv41_separator_padding("\ntext", true));
    t.assert_equal("complete leading separator", 0, mtmd_dsv41_separator_padding("\n\ntext", true));

    const std::string bar = "\xef\xbd\x9c";
    const std::string user = "<" + bar + "User" + bar + ">";
    const std::string assistant = "<" + bar + "Assistant" + bar + ">";
    t.assert_true("image-only user prefix", mtmd_dsv41_is_message_boundary(user, false));
    t.assert_true("assistant suffix", mtmd_dsv41_is_message_boundary(assistant + "</think>", true));
    t.assert_true("plain user text", !mtmd_dsv41_is_message_boundary("inspect", false));
}

//
// mtmd temporal merge
//

MAKE_TEST(test_temporal_merge_grouping) {
    std::vector<mtmd::bitmap_ptr> pool; // keeps the bitmaps alive until the end of the test

    // spec chars:
    //   v = video frame, w = video frame of another size, a = audio, i = plain image, t = text
    auto make_parts = [&pool](const std::string & spec) {
        std::vector<mtmd_internal_part> parts;
        for (char c : spec) {
            if (c == 't') {
                parts.push_back({ "hello", nullptr });
                continue;
            }
            mtmd_bitmap * bm = nullptr;
            switch (c) {
                case 'v': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                case 'w': bm = mtmd_bitmap_init(200, 200, nullptr);   break;
                case 'a': bm = mtmd_bitmap_init_from_audio(100, nullptr); break;
                case 'i': bm = mtmd_bitmap_init(100, 100, nullptr);   break;
                default: throw std::runtime_error(std::string("unknown spec char: ") + c);
            }
            mtmd_bitmap_set_mergeable(bm, c != 'i');
            pool.emplace_back(bm);
            parts.push_back({ "", bm });
        }
        return parts;
    };

    // { parts, n_merge, expected size of each group }
    const std::vector<std::tuple<std::string, int, std::string>> cases = {
        { "vv",   2, "2"    },
        { "vvv",  2, "21"   },
        { "vvvv", 2, "22"   },
        { "vvi",  2, "21"   },
        { "tvvt", 2, "2"    },
        { "vtv",  2, "11"   }, // text in between breaks the merge
        { "vw",   2, "11"   }, // different sizes cannot be merged
        { "aa",   2, "11"   }, // audio is never merged
        { "ii",   2, "11"   }, // two unrelated images must stay separated
        { "iv",   2, "11"   },
        { "vi",   2, "11"   },
        { "vv",   1, "11"   }, // model without temporal merge
    };

    for (const auto & [spec, n_merge, expected] : cases) {
        auto parts  = make_parts(spec);
        auto groups = mtmd_group_mergeable_bitmaps(parts, n_merge);

        std::string actual;
        for (const auto & group : groups) {
            actual += std::to_string(group.size());
        }

        const std::string name = "\"" + spec + "\" with n_merge=" + std::to_string(n_merge);
        t.assert_equal("groups for " + name, expected, actual);

        size_t n_bitmap_parts = 0;
        for (const auto & p : parts) {
            n_bitmap_parts += p.bitmap != nullptr ? 1 : 0;
        }
        t.assert_equal("remaining bitmap parts for " + name, groups.size(), n_bitmap_parts);
    }
}

//
// main
//

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = true;

    // usage: test-mtmd-impl [filter_regex]
    for (int i = 1; i < argc; i++) {
        t.set_filter(argv[i]);
    }

    for (const auto & e : test_registry::all()) {
        t.test(e.name, e.fn);
    }

    return t.summary();
}
