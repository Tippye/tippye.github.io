hexo.extend.filter.register('theme_inject', function (injects) {
    injects.head.raw('default', '<script src="/live2d-widget/autoload.js"></script>');
});