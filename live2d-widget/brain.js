! function() {
    function addBrain() {
        console.log("看板娘正在长出脑子~")
        const dialog = document.getElementById("waifu-tool-hitokoto"),
            dialog_container = document.createElement("div")

        dialog_container.innerHTML = `<div id='dialog-container' style='display: none;position: fixed;left:400px;bottom:50px;z-index:99'><input id='user-input' /><button id='submit-btn'>发送</button></div>`
        document.getElementsByTagName('body')[0].appendChild(dialog_container)
        dialog.addEventListener("click", function() {
            document.getElementById("dialog-container").style.display = "block";
        });
        document.getElementById("submit-btn").addEventListener("click", function() {
            var userInput = document.getElementById("user-input").value;
            if (userInput.trim() !== "") {
                // 这里可以添加你的回答逻辑，例如调用一个API或者使用预定义的回答
                var response = getResponse(userInput);
                showMessage(response, 6000, 9);
            }
            document.getElementById("dialog-container").style.display = "none";
        });
    }

    function getResponse(input) {
        // 简单示例：根据用户输入返回预定义的回答
        var responses = {
            "你好": "你好！很高兴见到你。",
            "你是谁": "我是你的看板娘助手。",
            "再见": "再见！希望很快能再见到你。"
        };
        return responses[input] || "对不起，我不明白你的意思。";
    }

    function showMessage(text, timeout, priority) {
        if (!text) return;
        if (!timeout) timeout = 5000;
        if (!priority) priority = 0;
        var waifuTips = document.getElementById("waifu-tips");
        waifuTips.innerHTML = text;
        waifuTips.classList.add("waifu-tips-active");
        setTimeout(function() {
            waifuTips.classList.remove("waifu-tips-active");
        }, timeout);
    }

    window.onload = function() {
        addBrain()
    }
}()