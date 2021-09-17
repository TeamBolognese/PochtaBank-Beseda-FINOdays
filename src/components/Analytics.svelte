
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	import Assessment from "./Assessment.svelte";
	import Ratio from "./Ratio.svelte";
	import Actions from "./Actions.svelte";
	
	// [ IMPORTS: props ]
	export let assessment = 'positive';
	export let ratio = {
        positive: 0,
        neutral: 0,
        negative: 0,
    };
	export let actions = [];


</script>


<!-- [ TEMPLATE: Analytics > Dashboard > Root ] -->
<div class = 'section-analytics'>

	<!-- [ section: emotional backgroud ] -->
	<Assessment { assessment } />
	
	<!-- [ section: emotional backgrounds ratio ] -->
	<Ratio { ratio } />
	
	<!-- [ section: action suggestions ] -->
	<Actions { actions } />

</div>