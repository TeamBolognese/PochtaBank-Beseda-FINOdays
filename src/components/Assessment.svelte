
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	import Emoji from "./Emoji.svelte";
	
	// [ IMPORTS: props ]
	export let assessment = 'positive';

</script>


<!-- [ TEMPLATE: Assessment > Analytics > Dashboard ] -->
<div class = 'section-assessment'>

	<h2 class = 'section-title'>Эмоциональный фон</h2>

    <div class="content">
        {#if assessment == 'positive'}
            <p class = 'txt-green'>Положительный</p>
            <Emoji type = 'positive' />
        {:else if assessment == 'neutral'}
            <p class = 'txt-yellow'>Нейтральный</p>
            <Emoji type = 'neutral' />
        {:else}
            <p class = 'txt-red'>Негативный</p>
            <Emoji type = 'negative' />
        {/if}
    </div>

</div>