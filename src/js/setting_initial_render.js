function setting_initial_render() {
    var gpu_graph = document.getElementById("gpu_graph");

    var meters = gpu_graph.getElementsByTagName('meter');
    var dls = gpu_graph.getElementsByTagName('dl');

    for(var i=0; i<meters.length; i++) {
        var parent = meters[i].parentElement;
        parent.insertBefore(dls[i], meters[i]);
    }
}