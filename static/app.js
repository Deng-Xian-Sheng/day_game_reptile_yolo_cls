/*
 * 搜索与排序逻辑。
 *
 * 搜索语法：
 *   - 多个包含词：空格、逗号、*、“和”、“与”、“及”分隔，需同时包含
 *   - 排除词：-词、!词，或不包含/不含/排除 + 词
 *   - 包含和排除可以混合使用，例如：
 *       动漫 护士 -口交
 *       包含动漫和护士 不包含口交
 *       不包含A和B 包含C
 *
 * 排序：
 *   - 可添加多个排序条件，按标签顺序逐级比较
 *   - 点击排序标签切换升序/降序，点 × 移除
 *   - 支持发布时间、总图片数、命中图片数、其他图片数、最高置信度
 */
(function (global) {
  "use strict";

  var NEG_MARK = "\u0001NEG\u0001";
  var POS_MARK = "\u0001POS\u0001";
  var SPLIT_RE = /(\u0001NEG\u0001|\u0001POS\u0001|[\s,;+*，、；和与及]+)/;
  var SEPARATOR_RE = /^[\s,;+*，、；和与及]+$/;

  var FIELD_META = {
    published: { label: "发布时间", type: "date" },
    total: { label: "总图片数", type: "number" },
    matched: { label: "命中图片数", type: "number" },
    other: { label: "其他图片数", type: "number" },
    confidence: { label: "最高置信度", type: "number" }
  };

  /**
   * 解析搜索框内容。
   * 返回 { include: ["词1", "词2"], exclude: ["词3"] }
   */
  function parseQuery(raw) {
    var include = [];
    var exclude = [];
    if (!raw) return { include: include, exclude: exclude };

    var text = String(raw)
      .replace(/[！]/g, "!")
      .replace(/[，、；]/g, ",")
      .replace(/[－—–]/g, "-");

    // 引号里的内容视为一个完整词组，不按空格/“和”拆分，
    // 也避免词组里的“包含/排除”被误识别。
    var quoted = [];
    text = text.replace(/["“”]([^"“”]+)["“”]/g, function (_match, phrase) {
      var token = "\u0002Q" + quoted.length + "\u0002";
      quoted.push(phrase.trim());
      return " " + token + " ";
    });

    text = text
      // 注意：先替换排除词，避免“不包含”被当成“包含”
      .replace(/不包含|不含|排除/g, " " + NEG_MARK + " ")
      .replace(/包含/g, " " + POS_MARK + " ");

    var mode = "include";
    var pendingExclude = false;
    var seenInclude = Object.create(null);
    var seenExclude = Object.create(null);
    var parts = text.split(SPLIT_RE);

    parts.forEach(function (part) {
      if (!part || SEPARATOR_RE.test(part)) return;
      if (part === NEG_MARK) {
        mode = "exclude";
        pendingExclude = false;
        return;
      }
      if (part === POS_MARK) {
        mode = "include";
        pendingExclude = false;
        return;
      }
      // 处理 "- " 与引号词组分开的情况：- "词组"
      if (part === "-" || part === "!") {
        pendingExclude = true;
        return;
      }

      var term = part.trim().replace(/^[:：]+/, "");
      if (!term) return;

      var isExclude = mode === "exclude" || pendingExclude;
      pendingExclude = false;
      if (term.charAt(0) === "-" || term.charAt(0) === "!") {
        isExclude = true;
        term = term.slice(1).trim();
      }
      if (!term) return;

      var quoteMatch = term.match(/^\u0002Q(\d+)\u0002$/);
      if (quoteMatch) {
        term = quoted[Number(quoteMatch[1])] || "";
      } else {
        term = term.replace(/^[:：]+/, "");
      }
      if (!term) return;
      term = term.toLowerCase();

      if (isExclude) {
        if (!seenExclude[term]) {
          seenExclude[term] = true;
          exclude.push(term);
        }
      } else if (!seenInclude[term]) {
        seenInclude[term] = true;
        include.push(term);
      }
    });

    return { include: include, exclude: exclude };
  }

  function matchesQuery(text, parsed) {
    var haystack = String(text || "").toLowerCase();
    var i;
    for (i = 0; i < parsed.include.length; i += 1) {
      if (haystack.indexOf(parsed.include[i]) === -1) return false;
    }
    for (i = 0; i < parsed.exclude.length; i += 1) {
      if (haystack.indexOf(parsed.exclude[i]) !== -1) return false;
    }
    return true;
  }

  function renderQueryPreview(parsed) {
    var box = document.getElementById("query-preview");
    if (!box) return;
    box.textContent = "";
    if (!parsed.include.length && !parsed.exclude.length) {
      box.hidden = true;
      return;
    }

    function addGroup(label, terms, className) {
      if (!terms.length) return;
      var labelEl = document.createElement("span");
      labelEl.className = "query-label";
      labelEl.textContent = label;
      box.appendChild(labelEl);
      terms.forEach(function (term) {
        var tag = document.createElement("span");
        tag.className = "query-tag " + className;
        tag.textContent = term;
        box.appendChild(tag);
      });
    }

    addGroup("包含", parsed.include, "include");
    addGroup("排除", parsed.exclude, "exclude");
    box.hidden = false;
  }

  function getSortValue(card, field) {
    if (field === "published") {
      var raw = card.getAttribute("data-published") || "";
      if (!raw) return null;
      var timestamp = Date.parse(raw);
      return Number.isNaN(timestamp) ? null : timestamp;
    }
    if (field === "confidence") {
      return parseFloat(card.getAttribute("data-confidence") || "0") || 0;
    }
    if (field === "total") {
      return parseInt(card.getAttribute("data-total") || "0", 10) || 0;
    }
    if (field === "matched") {
      return parseInt(card.getAttribute("data-matched") || "0", 10) || 0;
    }
    if (field === "other") {
      return parseInt(card.getAttribute("data-other") || "0", 10) || 0;
    }
    return 0;
  }

  function compareValues(a, b, direction) {
    // 缺失值永远排在最后
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    if (a === b) return 0;
    var result = a < b ? -1 : 1;
    return direction === "asc" ? result : -result;
  }

  function compareCards(a, b, rules) {
    for (var i = 0; i < rules.length; i += 1) {
      var rule = rules[i];
      var compared = compareValues(
        getSortValue(a, rule.field),
        getSortValue(b, rule.field),
        rule.dir
      );
      if (compared !== 0) return compared;
    }
    // 所有排序条件都相同时，保持服务器原始顺序
    var ai = parseInt(a.getAttribute("data-index") || "0", 10) || 0;
    var bi = parseInt(b.getAttribute("data-index") || "0", 10) || 0;
    return ai - bi;
  }

  function init() {
    // ------------------------------------------------------------------
    // 图片加载失败占位
    // ------------------------------------------------------------------
    document.addEventListener("error", function (event) {
      var img = event.target;
      if (img && img.tagName === "IMG") {
        var box = img.closest(".media-button, .strip-button, .thumb-button");
        if (box) box.classList.add("broken");
      }
    }, true);

    // ------------------------------------------------------------------
    // 灯箱
    // ------------------------------------------------------------------
    var lightbox = document.getElementById("lightbox");
    var lightboxImg = document.getElementById("lightbox-img");

    function closeLightbox() {
      if (!lightbox) return;
      lightbox.hidden = true;
      if (lightboxImg) lightboxImg.removeAttribute("src");
      document.body.classList.remove("no-scroll");
    }

    document.addEventListener("click", function (event) {
      var trigger = event.target.closest("[data-lightbox]");
      if (trigger && lightbox && lightboxImg) {
        event.preventDefault();
        lightboxImg.src = trigger.getAttribute("data-lightbox");
        lightbox.hidden = false;
        document.body.classList.add("no-scroll");
        return;
      }
      if (lightbox && !lightbox.hidden &&
          (event.target === lightbox || event.target.closest(".lightbox-close"))) {
        closeLightbox();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && lightbox && !lightbox.hidden) closeLightbox();
    });

    // ------------------------------------------------------------------
    // 搜索 + 排序
    // ------------------------------------------------------------------
    var filter = document.getElementById("filter");
    var cardsContainer = document.getElementById("cards");
    var cards = Array.prototype.slice.call(document.querySelectorAll(".card"));
    if (!filter || !cardsContainer || !cards.length) return;

    var shown = document.getElementById("shown");
    var empty = document.getElementById("empty-filter");
    var sortField = document.getElementById("sort-field");
    var sortDir = document.getElementById("sort-dir");
    var sortAdd = document.getElementById("sort-add");
    var sortClear = document.getElementById("sort-clear");
    var sortRulesBox = document.getElementById("sort-rules");

    // 初始不设置排序，保持 matched.json 的原始顺序（分类结果默认按置信度降序）。
    // 用户添加的第一个排序条件就是最高优先级，避免默认条件抢占第一优先级。
    var sortRules = [];

    function renderSortRules() {
      if (!sortRulesBox) return;
      sortRulesBox.textContent = "";
      if (!sortRules.length) {
        var emptyEl = document.createElement("span");
        emptyEl.className = "sort-empty";
        emptyEl.textContent = "未设置排序（保持原始顺序）";
        sortRulesBox.appendChild(emptyEl);
        return;
      }

      sortRules.forEach(function (rule, index) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sort-chip";
        chip.setAttribute("data-rule-index", String(index));
        chip.title = "点击切换升序/降序";

        var order = document.createElement("span");
        order.className = "sort-order";
        order.textContent = String(index + 1);

        var label = document.createElement("span");
        label.textContent = FIELD_META[rule.field] ? FIELD_META[rule.field].label : rule.field;

        var arrow = document.createElement("span");
        arrow.className = "sort-arrow";
        arrow.textContent = rule.dir === "asc" ? "↑" : "↓";

        var remove = document.createElement("span");
        remove.className = "sort-remove";
        remove.setAttribute("data-remove", String(index));
        remove.title = "移除该排序";
        remove.textContent = "×";

        chip.appendChild(order);
        chip.appendChild(label);
        chip.appendChild(arrow);
        chip.appendChild(remove);
        sortRulesBox.appendChild(chip);
      });
    }

    function applySort() {
      var sorted = cards.slice().sort(function (a, b) {
        return compareCards(a, b, sortRules);
      });
      sorted.forEach(function (card) {
        cardsContainer.appendChild(card);
      });
    }

    function applyFilter() {
      var parsed = parseQuery(filter.value);
      renderQueryPreview(parsed);
      var visible = 0;
      cards.forEach(function (card) {
        var ok = matchesQuery(card.getAttribute("data-search"), parsed);
        card.hidden = !ok;
        if (ok) visible += 1;
      });
      if (shown) shown.textContent = "显示 " + visible + " / " + cards.length + " 篇";
      if (empty) empty.hidden = visible !== 0;
    }

    filter.addEventListener("input", applyFilter);

    if (sortRulesBox) {
      sortRulesBox.addEventListener("click", function (event) {
        var remove = event.target.closest("[data-remove]");
        if (remove) {
          event.preventDefault();
          event.stopPropagation();
          var removeIndex = parseInt(remove.getAttribute("data-remove"), 10);
          if (!Number.isNaN(removeIndex)) {
            sortRules.splice(removeIndex, 1);
            renderSortRules();
            applySort();
          }
          return;
        }

        var chip = event.target.closest("[data-rule-index]");
        if (!chip) return;
        var index = parseInt(chip.getAttribute("data-rule-index"), 10);
        if (Number.isNaN(index) || !sortRules[index]) return;
        sortRules[index].dir = sortRules[index].dir === "asc" ? "desc" : "asc";
        renderSortRules();
        applySort();
      });
    }

    if (sortAdd && sortField && sortDir) {
      sortAdd.addEventListener("click", function () {
        var field = sortField.value;
        var direction = sortDir.value === "asc" ? "asc" : "desc";
        var existing = null;
        for (var i = 0; i < sortRules.length; i += 1) {
          if (sortRules[i].field === field) {
            existing = sortRules[i];
            break;
          }
        }
        if (existing) {
          existing.dir = direction;
        } else {
          sortRules.push({ field: field, dir: direction });
        }
        renderSortRules();
        applySort();
      });
    }

    if (sortClear) {
      sortClear.addEventListener("click", function () {
        sortRules = [];
        renderSortRules();
        applySort();
      });
    }

    renderSortRules();
    applySort();
    applyFilter();
  }

  // 浏览器中启动；Node 中只导出解析/比较函数，便于测试
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      parseQuery: parseQuery,
      matchesQuery: matchesQuery,
      getSortValue: getSortValue,
      compareCards: compareCards
    };
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
