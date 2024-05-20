function input_initial_render() {
    var output_box = document.getElementById("output_box");
    var output_box_wrap = output_box.getElementsByClassName("wrap")[0];
    var svelte_class_nm = Array.from(output_box_wrap.classList).find(cls => cls.startsWith('svelte'));

    var new_element = document.createElement('div');
    new_element.classList.add(svelte_class_nm);
    new_element.classList.add("meta-text");
    new_element.style.paddingRight = '30px';
    output_box_wrap.appendChild(new_element);
}