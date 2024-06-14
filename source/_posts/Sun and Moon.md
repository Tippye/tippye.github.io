---
title: Sun and Moon
date: 2022-03-10T22:45:55+08:00
categories: 大前端
tags:
    - 特效
mermaid: true
---
web 端的夜间模式终于玩明白了

要么给顶层标签加个标记，页面标签对应写样式

要么在 head 里加个 style 标签，里面改全局变量

## 原理

```flowchart
st=>start: 页面加载
cond=>condition: 检查本地缓存(localStorage)
op1=>operation: 没有本地缓存记录（夜间模式状态）
op2=>operation: 有本地缓存记录
op3=>operation: 读取系统主题状态
op4=>operation: 读取本地缓存
op5=>operation: 点击主题切换按钮
op6=>operation: 监听系统主题变化
e=>end: 设置主题

st->cond
cond(yes)->op2->op4->e
cond(no)->op1->op3->e

```

# 原生实现

```html
<!-- aria属性用来做无障碍 -->
<button class="theme-toggle" id="theme-toggle" aria-label="auto" aria-live="polite">
  <svg class="sun-and-moon" aria-hidden="true" width="24" height="24" viewBox="0 0 24 24">
    <!-- 这是太阳中间的圆 -->
    <circle class="sun" cx="12" cy="12" r="6" mask="url(#moon-mask)" fill="currentColor" />
    <!-- 这是太阳一圈的阳光射线 -->
    <g class="sun-beams" stroke="currentColor">
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </g>
    <!-- 这是一个遮罩层，用来遮住太阳的圆变成月亮，参考下面示意图 -->
    <mask class="moon" id="moon-mask">
      <rect x="0" y="0" width="100%" height="100%" fill="white" />
      <circle cx="24" cy="10" r="6" fill="black" />
    </mask>
  </svg>
</button>
```

```css
.theme-toggle {
  --size: 2rem;
  
  background: none;
  border: none;
  padding: 0;

  inline-size: var(--size);
  block-size: var(--size);
  aspect-ratio: 1;
  border-radius: 50%;

  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
  outline-offset: 5px;

  /* 对于无鼠标的用户，增大图标 */
  @media (hover: none) {
    --size: 48px;
  }
}
.theme-toggle > svg {
  inline-size: 100%;
  block-size: 100%;
  stroke-linecap: round;
}
.sun-and-moon > :is(.moon, .sun, .sun-beams){
  transform-origin: center center;
}
.sun-and-moon > :is(.moon, .sun){
  fill: var(--icon-fill);
}
.sun-and-moon > .sum-beams{
  stroke: var(--icon-fill);
  stroke-width: 2px;
}
.theme-toggle:is(:hover, :focus-visible) > (.sun-and-moon > :is(.moon, .sun)){
 fill: var(--icon-fill-hover);
}
.theme-toggle:is(:hover, :focus-visible) (.sun-and-moon > .sun-beams){
  stroke: var(--icon-fill-hover);
}

.sun-and-moon[data-theme="dark"] > .sun{
  transform: scale(1.75);
  transition-timing-function: cubic-bezier(.25,0,.3,1);
  transition-duration: .25s;
}
.sun-and-moon[data-theme="dark"] > .sun-beams{
  opacity: 0;
  transform: rotateZ(-25deg);
  transition-duration: .15s;
}
.sun-and-moon[data-theme="dark"] > .moon > circle{
  transform: translateX(-7px);

  @supports (cx: 1) {
    transform: translateX(0);
    cx: 17;
  }
}


.sun-and-moon > .sun {
  transition: transform .5s cubic-bezier(.5,1.25,.75,1.25);
}
.sun-and-moon > .sun-beams{
  transition: 
    transform .5s var(--ease-elastic-4),
    opacity .5s var(--ease-3)
  ;
}
```

```javascript
const storageKey = 'theme-preference'

const getColorPreference = () => {
  if (localStorage.getItem(storageKey))
    return localStorage.getItem(storageKey)
  else
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
}

const setPreference = () => {
  localStorage.setItem(storageKey, theme.value)
  reflectPreference()
}

const reflectPreference = () => {
  document.firstElementChild
    .setAttribute('data-theme', theme.value)

  document
    .querySelector('#theme-toggle')
    ?.setAttribute('aria-label', theme.value)
}

const onClick = () => {
  theme.value = theme.value === 'light'
    ? 'dark'
    : 'light'

  setPreference()
}

window.onload = () => {
  // set on load so screen readers can get the latest value on the button
  reflectPreference()

  // now this script can find and listen for clicks on the control
  document
    .querySelector('#theme-toggle')
    .addEventListener('click', onClick)
}
//同步系统主题
window
  .matchMedia('(prefers-color-scheme: dark)')
  .addEventListener('change', ({matches:isDark}) => {
    theme.value = isDark ? 'dark' : 'light'
    setPreference()
  })
```

![svg原理图](http://static.tippy.icu/blogImg/eHGsxT.jpg)

> 此按钮样式转载于[Adam Argyle的博客](https://web.dev/building-a-theme-switch-component/)

# Vue3实现

```javascript
<template>
...
<button aria-label="auto" aria-live="polite" class="relative sun-and-moon-box" @click="handleSetDarkMode">
  <svg aria-hidden="true" class="sun-and-moon rounded-full" height="24" viewBox="0 0 24 24"
        width="24">
    <circle class="sun text-yellow-300 shadow-sm shadow-amber-300 dark:text-gray-100" cx="12" cy="12"
            fill="currentColor"
            mask="url(#moon-mask)"
            r="6"/>
    <g class="sun-beams text-yellow-300 dark:text-gray-100" stroke="currentColor">
      <line x1="12" x2="12" y1="1" y2="3"/>
      <line x1="12" x2="12" y1="21" y2="23"/>
      <line x1="4.22" x2="5.64" y1="4.22" y2="5.64"/>
      <line x1="18.36" x2="19.78" y1="18.36" y2="19.78"/>
      <line x1="1" x2="3" y1="12" y2="12"/>
      <line x1="21" x2="23" y1="12" y2="12"/>
      <line x1="4.22" x2="5.64" y1="19.78" y2="18.36"/>
      <line x1="18.36" x2="19.78" y1="5.64" y2="4.22"/>
    </g>
    <mask id="moon-mask" class="moon text-gray-darkest dark:text-gray-100">
      <rect fill="white" height="100%" width="100%" x="0" y="0"/>
      <circle cx="24" cy="10" fill="black" r="6"/>
    </mask>
  </svg>
</button>
...
</template>

<script setup>
import store from "@/store";
import {computed, inject} from "vue";
...
let darkMode = computed(() => store.state.settings.darkMode)//这里的值是布尔值
const handleSetDarkMode = () => {
  console.log('handleSetDarkMode')
  store.dispatch('settings/toggleDarkMode', {
    value: !darkMode.value
  })
}
...
</script>
<style lang="scss" scoped>
.sun-and-moon-box {
  width: 24px;
  height: 24px;
}

.sun-and-moon {
  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
  outline-offset: 5px;

  & > svg {
    inline-size: 100%;
    block-size: 100%;
    stroke-linecap: round;
  }

  & > * {
    transform-origin: center center;;
  }

  & > .sun {
    transition: transform .5s cubic-bezier(.5, 1.25, .75, 1.25);
  }

  & > .sun-beams {
    transition: transform .5s cubic-bezier(.5, 1.5, .75, 1.25),
    opacity .5s cubic-bezier(.25, 0, .3, 1);

    & > line {
      color: inherit;
    }
  }

  & > .moon > circle {
    transition-delay: .25s;
    transition-duration: .5s;
  }
}
/* 这里的dark是html的类名，切换成深色模式时会给html添加dark类名 */
.dark .sun-and-moon {
  & > .sun {
    transform: scale(1.5);
    transition-timing-function: cubic-bezier(.25, 0, .3, 1);
    transition-duration: .25s;
  }

  & > .sun-beams {
    opacity: 0;
    transition-duration: .15s
  }

  & > .moon > circle {
    transform: translateX(-7px);
    transition: transform .25s cubic-bezier(0, 0, 0, 1);

    @supports (cx: 1) {
      transform: translateX(0);
      cx: 17;
      transition: cx .25s cubic-bezier(0, 0, 0, 1);
    }
  }
}
</style>
```

```javascript
//这是vuex的/store/modules/settings.vue
const state = {
...
  //夜间模式
  darkMode: false,
...
}

const mutations = {
    CHANGE_SETTING: (state, {key, value}) => {
        if (state.hasOwnProperty(key)) {
            state[key] = value
            Cookie.set(key, value)
        }
    }
}

const actions = {
    ...
    /**
     * 更改夜间模式
     *
     * @param commit
     * @param value{Boolean} true:夜间模式|false:默认主题
     * @return {Promise<void>}
     */
    async toggleDarkMode({commit}, {value}) {
        value ? document.documentElement.classList.add('dark') : document.documentElement.classList.remove('dark')
        console.log('toggleDarkMode')
        commit('CHANGE_SETTING', {key: 'darkMode', value})
    },
    ...
}

...
```
