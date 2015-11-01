var graph = new joint.dia.Graph();

var module_height = 100;
var module_width = 100;

var defaultLink = new joint.dia.Link(
{
	attrs:
	{
		'.marker-target': { d: 'M 10 0 L 0 5 L 10 10 z', },
		'.link-tools .tool-remove circle, .marker-vertex': { r: 8 },
	},
});

function validateConnection(cellViewS, magnetS, cellViewT, magnetT, end, linkView)
{
	if (magnetS && magnetS.getAttribute('type') === 'input') return false;

	if (magnetT && magnetT.getAttribute('type') === 'input') {
		var links = graph.getConnectedLinks(cellViewS.model);
		for (var i = 0; i < links.length; i++)
		{
			var link = links[i];
			if (link.attributes.source.id === cellViewS.model.id && link.attributes.source.port === magnetS.attributes.port.nodeValue && link.attributes.target.id)
			{
				var targetCell = graph.getCell(link.attributes.target.id);
				if (targetCell == cellViewT.model)
					return false; // Already connected
			} 
		}

		return true;
	}

	return false;
}

function validateMagnet(cellView, magnet)
{
	if (magnet.getAttribute('type') === 'input')
		return false;

	return true;
}

// override to position ports top and bottom
joint.shapes.devs.Model.prototype.getPortAttrs = function(portName, index, total, selector, type) {
    var attrs = {};
    
    var portClass = 'port' + index;
    var portSelector = selector + '>.' + portClass;
    var portLabelSelector = portSelector + '>.port-label';
    var portBodySelector = portSelector + '>.port-body';

    attrs[portLabelSelector] = { text: portName };
    attrs[portBodySelector] = { port: { id: portName || _.uniqueId(type) , type: type } };
    attrs[portSelector] = { ref: '.body', 'ref-x': (index + 0.5) * (1 / total) };
    
    if (selector === '.inPorts') { attrs[portSelector]['ref-dy'] = 0; }
    if (selector === '.outPorts') { attrs[portSelector]['ref-dy'] = -module_height; }

    return attrs;
}

joint.shapes.calm = {};
joint.shapes.calm.Base = joint.shapes.devs.Model.extend(
{
	defaults: joint.util.deepSupplement
	(
		{
			type: 'calm.Base',
			size: { width: module_width, height: module_height },
			name: '',
			attrs:
			{
				'.body': { stroke: 'none', 'fill-opacity': 0 },
				'.label': { display: 'none' },
				'.inPorts circle': { magnet: 'passive', type: 'input' },
				'.outPorts circle': { magnet: true, type: 'output' },
			},
		},
		joint.shapes.devs.Model.prototype.defaults
	),
});

joint.shapes.calm.BaseView = joint.shapes.devs.ModelView.extend(
{
	template:
	[
		'<div class="module">',
		'<span class="label"></span>',
		'<button class="delete">x</button>',
		'<input type="text" class="name" placeholder="Enter name" />',
        '<input type="text" class="size" placeholder="Enter size" />',
		'</div>',
	].join(''),

	initialize: function()
	{
		_.bindAll(this, 'updateBox');
		joint.shapes.devs.ModelView.prototype.initialize.apply(this, arguments);

		this.$box = $(_.template(this.template)());
		// Prevent paper from handling pointerdown.
		this.$box.find('input').on('mousedown click', function(evt) { evt.stopPropagation(); });

		// This is an example of reacting on the input change and storing the input data in the cell model.
		this.$box.find('input.name').on('change', _.bind(function(evt)
		{
			this.model.set('name', $(evt.target).val());
		}, this));

        this.$box.find('input.size').on('change', _.bind(function(evt)
        {
            this.model.set('mdl_size', $(evt.target).val());
        }, this));

		this.$box.find('.delete').on('click', _.bind(this.model.remove, this.model));
		// Update the box position whenever the underlying model changes.
		this.model.on('change', this.updateBox, this);
		// Remove the box when the model gets removed from the graph.
		this.model.on('remove', this.removeBox, this);

		this.updateBox();
	},

	render: function()
	{
		joint.shapes.devs.ModelView.prototype.render.apply(this, arguments);
		this.paper.$el.prepend(this.$box);
		this.updateBox();
		return this;
	},

	updateBox: function()
	{
		// Set the position and dimension of the box so that it covers the JointJS element.
		var bbox = this.model.getBBox();

		// Update the HTML with a data stored in the cell model.
		var nameField = this.$box.find('input.name');
		if (!nameField.is(':focus'))
			nameField.val(this.model.get('name'));

        var sizeField = this.$box.find('input.size');
        if (!sizeField.is(':focus'))
            sizeField.val(this.model.get('mdl_size'));

		var label = this.$box.find('.label');
		var type = this.model.get('type').slice('calm.'.length);
		label.text(type);
		label.attr('class', 'label ' + type);
		this.$box.css({ width: bbox.width, height: bbox.height, left: bbox.x, top: bbox.y, transform: 'rotate(' + (this.model.get('angle') || 0) + 'deg)' });
	},

	removeBox: function(evt)
	{
		this.$box.remove();
	},
});

joint.shapes.calm.Input = joint.shapes.devs.Model.extend(
{
	defaults: joint.util.deepSupplement
	(
		{
			type: 'calm.Input',
			outPorts: ['output'],
		},
		joint.shapes.calm.Base.prototype.defaults
	),
});
joint.shapes.calm.InputView = joint.shapes.calm.BaseView;

joint.shapes.calm.Standard = joint.shapes.devs.Model.extend(
{
	defaults: joint.util.deepSupplement
	(
		{
			type: 'calm.Standard',
			inPorts: ['input'],
			outPorts: ['output'],
		},
		joint.shapes.calm.Base.prototype.defaults
	),
});
joint.shapes.calm.StandardView = joint.shapes.calm.BaseView;

joint.shapes.calm.Map = joint.shapes.devs.Model.extend(
{
	defaults: joint.util.deepSupplement
	(
		{
			type: 'calm.Map',
			inPorts: ['input'],
			outPorts: ['output'],
		},
		joint.shapes.calm.Base.prototype.defaults
	),
});
joint.shapes.calm.MapView = joint.shapes.calm.BaseView;

function applyTextFields()
{
	$('input[type=text]').blur();
}

function add(constructor)
{
	return function()
	{
		var position = $('.contextMenuPlugin').position();
		var container = $('#paper')[0];
        var container_pos = $('#paper').position();
		var element = new constructor(
		{
			position: { x: position.left + container.scrollLeft, y: position.top - container_pos.top + 20 },
		});
		graph.addCells([element]);
	};
}

function clear()
{
	graph.clear();
}

// Browser stuff

var paper = new joint.dia.Paper(
{
	el: $('#paper'),
	width: 3200,
	height: 1800,
	model: graph,
	gridSize: 20,
	defaultLink: defaultLink,
	validateConnection: validateConnection,
	validateMagnet: validateMagnet,
	// Enable link snapping within 80px lookup radius
	snapLinks: { radius: 80 }
});

paper.on('cell:pointerdown', function(e, x, y)
{
	applyTextFields();
});

// don't keep links to nowhere
paper.on('cell:pointerup', function () {
    var links = paper.model.getLinks();
    _.each(links, function (link) {
        var source = link.get('source');
        var target = link.get('target');
        if (source.id === undefined || target.id === undefined) {
            link.remove();
        }
    });
});

function adjustVertices(graph, cell) {

    // If the cell is a view, find its model.
    cell = cell.model || cell;

    if (cell instanceof joint.dia.Element) {
    	var links = graph.getConnectedLinks(cell);
    	$.each(links, function( index, cell ) {
	        adjustVertices(graph, cell);
        });
        return;
    }

    // The cell is a link. Let's find its source and target models.
    var srcId = cell.get('source').id || cell.previous('source').id;
    var trgId = cell.get('target').id || cell.previous('target').id;

    // If one of the ends is not a model, the link has no siblings.
    if (!srcId || !trgId) return;

    // We only care about recursive links
    if (srcId != trgId) {
    	var verts = cell.get('vertices');
    	if (verts && verts.length > 1) {
    		// when a link moves over self but connects to other module,
    		// clean up the vertices
    		cell.unset('vertices');
    	}
    	return;
	}

    // recalculate
    cell.unset('vertices');

    var srcCenter = graph.getCell(srcId).getBBox().center();
    cell.set('vertices', [
    	{ x: srcCenter.x, y: srcCenter.y - 60 },
    	{ x: srcCenter.x + 120, y: srcCenter.y - 60 },
    	{ x: srcCenter.x + 120, y: srcCenter.y + 60 },
    	{ x: srcCenter.x, y: srcCenter.y + 60 }
    ]);
};

// adjust vertices when a cell is removed or its source/target was changed
var myAdjustVertices = _.partial(adjustVertices, graph);
graph.on('add remove change:source change:target', myAdjustVertices);
// also when an user stops interacting with an element.
paper.on('cell:pointerup', myAdjustVertices);

$('#paper').contextPopup({
	title: 'Tool Menu',
	items:
	[
		{ label: 'Input', icon:'/static/img/abacus.png', action: add(joint.shapes.calm.Input) },
		{ label: 'Standard', icon:'/static/img/abacus.png', action: add(joint.shapes.calm.Standard) },
		{ label: 'Map', icon:'/static/img/abacus.png', action: add(joint.shapes.calm.Map) },
		// { label: 'Save', icon:'/static/img/abacus.png', action: save },
		// { label: 'Load', icon:'/static/img/abacus.png', action: load },
		// { label: 'Import', id: 'import', icon:'/static/img/abacus.png', action: importFile },
		// { label: 'New', icon:'/static/img/abacus.png', action: clear },
		// { label: 'Export', id: 'export', icon:'/static/img/abacus.png', action: exportFile },
	]
});

$("#network_form").submit(function(e) {
    if ($('#id_name').val().length == 0) {
        e.preventDefault();
        alert("Please enter a name for this network!");
        return false;
    }
    $('.module input').each(function( index ) {
        if ($(this).val().length == 0) {
            e.preventDefault();
            alert("Please enter names and sizes for each module!");
            return false;
        }
    });

    $('#id_definition').val(JSON.stringify(graph));
    return true;
});

$(document).ready(function() {
    var graph_data = $('#id_definition').val();
    if (graph_data.length > 0) {
        graph.fromJSON(JSON.parse(graph_data));
    }
});
